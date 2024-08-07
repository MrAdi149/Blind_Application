package com.aditya.`object`.fragment

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation
import com.aditya.`object`.BitmapUtils
import com.aditya.`object`.DetectionResult
import com.aditya.`object`.ObjectDetectorHelper
import com.aditya.`object`.R
import com.aditya.`object`.databinding.FragmentCameraBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.*
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    private val TAG = "ObjectDetection"
    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!
    private var isObjectDetection = false
    private var isTextRecognition = false
    private var textRecognizer: TextRecognizer? = null
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private var isListening = false
    private var describedDetection: Detection? = null

    private val RECORD_REQUEST_CODE = 101
    private val RESTART_DELAY_MS = 1000L  // 1 second delay before restarting speech recognizer

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var textToSpeech: TextToSpeech
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var previousDetectedObject: String? = null

    companion object {
        private const val KEY_IS_OBJECT_DETECTION = "KEY_IS_OBJECT_DETECTION"
        private const val KEY_IS_TEXT_RECOGNITION = "KEY_IS_TEXT_RECOGNITION"
        private const val ANALYSIS_INTERVAL_MS = 500L
        private var lastAnalyzedTimestamp = 0L
    }

    override fun onResume() {
        super.onResume()
        if (!PermissionFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.nav_host_fragment)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        } else {
            if (!isListening) {
                startListening()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        stopListening()
        speechRecognizer.destroy()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.Builder().build())
        checkPermission()
        initButtons()
        return fragmentCameraBinding.root
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(KEY_IS_OBJECT_DETECTION, isObjectDetection)
        outState.putBoolean(KEY_IS_TEXT_RECOGNITION, isTextRecognition)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if (savedInstanceState != null) {
            isObjectDetection = savedInstanceState.getBoolean(KEY_IS_OBJECT_DETECTION)
            isTextRecognition = savedInstanceState.getBoolean(KEY_IS_TEXT_RECOGNITION)
        }
        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )
        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status != TextToSpeech.ERROR) {
                textToSpeech.language = Locale.US
            }
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
        fragmentCameraBinding.viewFinder.post {
            setUpCamera()
        }
        initBottomSheetControls()
    }

    private fun checkPermission() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.RECORD_AUDIO),
                RECORD_REQUEST_CODE
            )
        } else {
            setupSpeechRecognizer()
        }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(requireContext())
        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isListening = true
            }

            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                isListening = false
            }

            override fun onError(error: Int) {
                Log.e(TAG, "Speech Recognition Error: $error")
                isListening = false
                restartSpeechRecognizer()
            }

            override fun onResults(results: Bundle?) {
                isListening = false
                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.let { matches ->
                    if (matches.isNotEmpty()) {
                        val command = matches[0].toLowerCase(Locale.ROOT)
                        handleVoiceCommand(command)
                    }
                }
                restartSpeechRecognizer()
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun restartSpeechRecognizer() {
        lifecycleScope.launch {
            delay(RESTART_DELAY_MS)
            if (isListening) {
                speechRecognizer.stopListening()
            }
            speechRecognizer.startListening(speechRecognizerIntent)
        }
    }

    private fun startListening() {
        if (!isListening) {
            speechRecognizer.startListening(speechRecognizerIntent)
        }
    }

    private fun stopListening() {
        if (isListening) {
            isListening = false
            speechRecognizer.stopListening()
        }
    }

    private fun handleVoiceCommand(command: String) {
        when (command) {
            "start" -> {
                isObjectDetection = true
                isTextRecognition = false
                describedDetection = null
                fragmentCameraBinding.overlay.clear()
                Toast.makeText(requireContext(), "Object Detection Started", Toast.LENGTH_SHORT).show()
            }
            "stop" -> {
                if (isObjectDetection) {
                    Log.d(TAG, "Object Detection Stopped")
                    Toast.makeText(requireContext(), "Object Detection Stopped", Toast.LENGTH_SHORT).show()
                } else if (isTextRecognition) {
                    Log.d(TAG, "Text Recognition Stopped")
                    Toast.makeText(requireContext(), "Text Recognition Stopped", Toast.LENGTH_SHORT).show()
                }
                isObjectDetection = false
                isTextRecognition = false
                fragmentCameraBinding.overlay.clear()
            }
            "read" -> {
                isObjectDetection = false
                isTextRecognition = true
                describedDetection = null
                fragmentCameraBinding.overlay.clear()
                Toast.makeText(requireContext(), "Text Recognition Started", Toast.LENGTH_SHORT).show()
            }
            "describe" -> {
                describedDetection?.let { detection ->
                    val description = getDescriptionForDetection(detection)
                    fragmentCameraBinding.overlay.describeObject(detection, description)
                    textToSpeech.speak(description, TextToSpeech.QUEUE_FLUSH, null, null)
                } ?: run {
                    Toast.makeText(requireContext(), "No object detected to describe.", Toast.LENGTH_SHORT).show()
                }
            }
            else -> {
                Toast.makeText(requireContext(), "Unknown Command: $command", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun getDescriptionForDetection(detection: Detection): String {
        val label = detection.categories[0].label
        val score = detection.categories[0].score
        val company = "Company X"
        val size = "medium"
        val color = "red"

        return "$label detected with $score confidence. Company: $company, Size: $size, Color: $color."
    }

    private fun initButtons() {
        fragmentCameraBinding.btnStartTextDetection.setOnClickListener {
            handleVoiceCommand("start")
        }
        fragmentCameraBinding.btnStopTextDetection.setOnClickListener {
            handleVoiceCommand("stop")
        }
        fragmentCameraBinding.btnStartTextDetection.setOnClickListener {
            handleVoiceCommand("read")
        }
        fragmentCameraBinding.btnStartTextDetection.setOnClickListener {
            handleVoiceCommand("describe")
        }
    }

    private fun initBottomSheetControls() {
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus.setOnClickListener {
            if (objectDetectorHelper.threshold >= 0.1) {
                objectDetectorHelper.threshold -= 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.thresholdPlus.setOnClickListener {
            if (objectDetectorHelper.threshold <= 0.8) {
                objectDetectorHelper.threshold += 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.maxResultsMinus.setOnClickListener {
            if (objectDetectorHelper.maxResults > 1) {
                objectDetectorHelper.maxResults--
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.maxResultsPlus.setOnClickListener {
            if (objectDetectorHelper.maxResults < 5) {
                objectDetectorHelper.maxResults++
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (objectDetectorHelper.numThreads > 1) {
                objectDetectorHelper.numThreads--
                updateControlsUi()
            }

        }
        fragmentCameraBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (objectDetectorHelper.numThreads < 4) {
                objectDetectorHelper.numThreads++
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(0, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    objectDetectorHelper.currentDelegate = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {}
            }
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(0, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    objectDetectorHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {}
            }
    }

    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxResultsValue.text =
            objectDetectorHelper.maxResults.toString()
        fragmentCameraBinding.bottomSheetLayout.thresholdValue.text =
            String.format("%.2f", objectDetectorHelper.threshold)
        fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
            objectDetectorHelper.numThreads.toString()
        objectDetectorHelper.clearObjectDetector()
        fragmentCameraBinding.overlay.clear()
    }

    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
        val cameraSelector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { image ->
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastAnalyzedTimestamp >= ANALYSIS_INTERVAL_MS) {
                        lastAnalyzedTimestamp = currentTime
                        when {
                            isObjectDetection -> detectObjects(image)
                            isTextRecognition -> detectText(image)
                            else -> image.close()
                        }
                    } else {
                        image.close()
                    }
                }
            }
        cameraProvider.unbindAll()
        try {
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectObjects(image: ImageProxy) {
        val bitmap = BitmapUtils.imageProxyToBitmap(image)
        val imageRotation = image.imageInfo.rotationDegrees
        image.close()
        lifecycleScope.launch(Dispatchers.Main) {
            objectDetectorHelper.detect(bitmap, imageRotation)
        }
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun detectText(image: ImageProxy) {
        val mediaImage = image.image ?: return
        val inputImage = InputImage.fromMediaImage(mediaImage, image.imageInfo.rotationDegrees)
        textRecognizer?.process(inputImage)
            ?.addOnSuccessListener { visionText ->
                fragmentCameraBinding.overlay.setResults(
                    emptyList(), mediaImage.height, mediaImage.width, visionText.textBlocks
                )
                if (visionText.textBlocks.isNotEmpty()) {
                    readDetectedText(visionText.textBlocks.map { it.text })
                }
                image.close()
            }
            ?.addOnFailureListener { e ->
                Log.e(TAG, "Text Recognition Error: $e")
                image.close()
            }
    }

    private fun readDetectedText(textBlocks: List<String>) {
        if (textBlocks.isEmpty() || textToSpeech.isSpeaking) {
            return
        }
        val textToRead = textBlocks.joinToString(" ")
        val utteranceId = "UtteranceID-${System.currentTimeMillis()}"
        textToSpeech.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) {
                // No-op
            }

            override fun onDone(utteranceId: String) {
                // Reset the textToSpeech state after completion
            }

            override fun onError(utteranceId: String) {
                // Handle error
            }
        })
        textToSpeech.speak(textToRead, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    override fun onDestroyView() {
        super.onDestroyView()
        textRecognizer?.close()

        textToSpeech.shutdown()
        cameraExecutor.shutdown()
        stopObjectDetection()
        _fragmentCameraBinding = null

        speechRecognizer.destroy()
    }

    private fun stopObjectDetection() {
        isObjectDetection = false
        isTextRecognition = false
        fragmentCameraBinding.overlay.clear()
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long, imageHeight: Int, imageWidth: Int
    ) {
        activity?.runOnUiThread {
            if (!results.isNullOrEmpty()) {
                val detectionResults = results.map { detection ->
                    // Assuming an object real height, you may need to adjust this
                    val objectRealHeight = 0.2f  // meters, for example
                    val distance = objectDetectorHelper.estimateDistance(detection.boundingBox, objectRealHeight)
                    DetectionResult(detection, distance)
                }

                val currentDetectedObject = detectionResults.firstOrNull()?.detection?.categories?.firstOrNull()?.label
                currentDetectedObject?.let { objectLabel ->
                    if (currentDetectedObject != previousDetectedObject) {
                        textToSpeech.speak(objectLabel, TextToSpeech.QUEUE_FLUSH, null, null)
                        previousDetectedObject = objectLabel
                    }
                }

                fragmentCameraBinding.overlay.setResults(
                    detectionResults, imageHeight, imageWidth
                )
                fragmentCameraBinding.overlay.invalidate()
            }
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)
        }
    }


    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}