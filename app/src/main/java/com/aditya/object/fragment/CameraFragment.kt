package com.aditya.`object`.fragment

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.SurfaceTexture
import android.hardware.usb.UsbDevice
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.LayoutInflater
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.camera.core.*
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.aditya.`object`.BitmapUtils
import com.aditya.`object`.ObjectDetectorHelper
import com.aditya.`object`.R
import com.aditya.`object`.databinding.FragmentCameraBinding
import com.jiangdg.usb.USBMonitor
import com.jiangdg.uvc.UVCCamera
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    private val TAG = "ObjectDetection"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private var isObjectDetection: Boolean = false
    private var isTextRecognition: Boolean = false
    private var textRecognizer: TextRecognizer? = null
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private var isListening: Boolean = false
    private var isWaitingForCommand: Boolean = false
    private var describedDetection: Detection? = null

    private lateinit var usbMonitor: USBMonitor
    private lateinit var uvcCamera: UVCCamera
    private var isUsingUsbCamera: Boolean = false

    private val RECORD_REQUEST_CODE = 101
    private val RESTART_DELAY_MS = 1000L  // 1 second delay before restarting speech recognizer

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
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
            startListening()
        }
    }

    override fun onPause() {
        super.onPause()
        stopListening()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.Builder().build())

        checkPermission()
        initButtons()

        return fragmentCameraBinding.root
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

        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Attach listeners to UI control widgets
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
        Handler(Looper.getMainLooper()).postDelayed({
            if (isListening) {
                speechRecognizer.stopListening()
            }
            speechRecognizer.startListening(speechRecognizerIntent)
        }, RESTART_DELAY_MS)
    }

    private fun startListening() {
        if (!isListening) {
            speechRecognizer.startListening(speechRecognizerIntent)
        }
    }

    private fun stopListening() {
        if (isListening) {
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
        // Simulate a description based on detection properties. Adjust as needed.
        val label = detection.categories[0].label
        val score = detection.categories[0].score
        val company = "Company X" // Replace with actual company info if available
        val size = "medium" // Replace with actual size info if available
        val color = "red" // Replace with actual color info if available

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
        // When clicked, lower detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus.setOnClickListener {
            if (objectDetectorHelper.threshold >= 0.1) {
                objectDetectorHelper.threshold -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdPlus.setOnClickListener {
            if (objectDetectorHelper.threshold <= 0.8) {
                objectDetectorHelper.threshold += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsMinus.setOnClickListener {
            if (objectDetectorHelper.maxResults > 1) {
                objectDetectorHelper.maxResults--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsPlus.setOnClickListener {
            if (objectDetectorHelper.maxResults < 5) {
                objectDetectorHelper.maxResults++
                updateControlsUi()
            }
        }

        // When clicked, decrease the number of threads used for detection
        fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (objectDetectorHelper.numThreads > 1) {
                objectDetectorHelper.numThreads--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of threads used for detection
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

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(0, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    objectDetectorHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset detector.
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
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
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
        // Convert YUV_420_888 ImageProxy to Bitmap
        val bitmap = BitmapUtils.imageProxyToBitmap(image)

        val imageRotation = image.imageInfo.rotationDegrees
        image.close()

        // Run detection on the main thread
        Handler(Looper.getMainLooper()).post {
            objectDetectorHelper.detect(bitmap, imageRotation)
        }
    }

    @OptIn(ExperimentalGetImage::class)
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
        speechRecognizer.stopListening()
        textToSpeech.shutdown()
        cameraExecutor.shutdown()
        _fragmentCameraBinding = null
    }


    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            if (!results.isNullOrEmpty()) {
                val currentDetectedObject = results[0].categories.firstOrNull()?.label

                currentDetectedObject?.let { objectLabel ->
                    if (currentDetectedObject != previousDetectedObject) {
                        textToSpeech.speak(objectLabel, TextToSpeech.QUEUE_FLUSH, null, null)
                        previousDetectedObject = objectLabel
                    }
                }
            }

            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList(),
                imageHeight,
                imageWidth
            )

            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}
