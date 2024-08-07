package com.aditya.`object`.fragment

import android.Manifest
import android.app.Activity
import android.app.Activity.RESULT_OK
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.location.Geocoder
import android.media.Image
import android.net.Uri
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.text.InputType
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.EditText
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation
import com.aditya.`object`.BitmapUtils
import com.aditya.`object`.DetectionResult
import com.aditya.`object`.ObjectDetectorHelper
import com.aditya.`object`.R
import com.aditya.`object`.SimilarityClassifier
import com.aditya.`object`.databinding.FragmentCameraBinding
import com.google.android.gms.location.LocationServices
import com.google.android.gms.maps.model.LatLng
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.ByteArrayOutputStream
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
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
    private val RESTART_DELAY_MS = 1000L
    private val REQUEST_LOCATION_PERMISSION = 102

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var textToSpeech: TextToSpeech
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var previousDetectedObject: String? = null

    private lateinit var broadcastReceiver: BroadcastReceiver

    // New variables for navigation
    private var isWaitingForDestination = false
    private var currentLatLng: LatLng? = null

    // Face Recognition Fields
    private lateinit var detector: FaceDetector
    private lateinit var tfLite: Interpreter
    private var registered = HashMap<String, SimilarityClassifier.Recognition>()
    private lateinit var previewView: PreviewView
    private var developerMode = false
    private var distance = 1.0f
    private var start = true
    private var flipX = false
    private var camFace = CameraSelector.LENS_FACING_BACK
    private lateinit var cameraSelector: CameraSelector
    private val inputSize = 112
    private var isModelQuantized = false
    private lateinit var embeddings: Array<FloatArray>
    private lateinit var intValues: IntArray
    private val imageMean = 128.0f
    private val imageStd = 128.0f
    private val outputSize = 192
    private val selectPicture = 1
    private val modelFile = "mobile_face_net.tflite"

    companion object {
        private const val KEY_IS_OBJECT_DETECTION = "KEY_IS_OBJECT_DETECTION"
        private const val KEY_IS_TEXT_RECOGNITION = "KEY_IS_TEXT_RECOGNITION"
        private const val ANALYSIS_INTERVAL_MS = 500L
        private const val NAVIGATION_REQUEST_CODE = 103
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
        initializeFaceRecognition()
    }

    private fun initializeFaceRecognition() {
        registered = readFromSP() // Load saved faces from shared preferences
        try {
            tfLite = Interpreter(loadModelFile(requireActivity(), modelFile))
        } catch (e: IOException) {
            e.printStackTrace()
        }

        val highAccuracyOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
        detector = FaceDetection.getClient(highAccuracyOpts)
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
            "navigation" -> {
                startNavigationSequence()
            }
            else -> {
                if (isWaitingForDestination) {
                    handleDestinationInput(command)
                } else {
                    Toast.makeText(requireContext(), "Unknown Command: $command", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun startNavigationSequence() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(requireActivity(), arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), REQUEST_LOCATION_PERMISSION)
        } else {
            getCurrentLocation()
        }
    }

    private fun getCurrentLocation() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_LOCATION_PERMISSION
            )
        } else {
            val fusedLocationClient = LocationServices.getFusedLocationProviderClient(requireContext())
            fusedLocationClient.lastLocation
                .addOnSuccessListener { location ->
                    if (location != null) {
                        currentLatLng = LatLng(location.latitude, location.longitude)
                        textToSpeech.speak("Please provide your destination", TextToSpeech.QUEUE_FLUSH, null, null)
                        isWaitingForDestination = true
                    } else {
                        Toast.makeText(requireContext(), "Unable to get current location. Make sure location services are enabled.", Toast.LENGTH_SHORT).show()
                    }
                }
                .addOnFailureListener {
                    Toast.makeText(requireContext(), "Failed to get current location", Toast.LENGTH_SHORT).show()
                }
        }
    }


    private fun handleDestinationInput(destination: String) {
        val geocoder = Geocoder(requireContext(), Locale.getDefault())
        try {
            val addresses = geocoder.getFromLocationName(destination, 1)
            if (addresses != null && addresses.isNotEmpty()) {
                val destinationLatLng = LatLng(addresses[0].latitude, addresses[0].longitude)
                openMapForNavigation(destinationLatLng)
            } else {
                Toast.makeText(requireContext(), "Destination not found.", Toast.LENGTH_SHORT).show()
            }
        } catch (e: IOException) {
            Toast.makeText(requireContext(), "Error finding destination: ${e.message}", Toast.LENGTH_SHORT).show()
        } finally {
            isWaitingForDestination = false
        }
    }

    private fun openMapForNavigation(destinationLatLng: LatLng) {
        val intent = Intent(requireContext(), MapsActivity::class.java).apply {
            putExtra("CURRENT_LAT", currentLatLng?.latitude)
            putExtra("CURRENT_LNG", currentLatLng?.longitude)
            putExtra("DESTINATION_LAT", destinationLatLng.latitude)
            putExtra("DESTINATION_LNG", destinationLatLng.longitude)
        }
        startActivityForResult(intent, NAVIGATION_REQUEST_CODE)
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

        fragmentCameraBinding.button5.setOnClickListener {
            camFace = if (camFace == CameraSelector.LENS_FACING_BACK) {
                flipX = true
                CameraSelector.LENS_FACING_FRONT
            } else {
                flipX = false
                CameraSelector.LENS_FACING_BACK
            }
            cameraProvider?.unbindAll()
            bindCameraUseCases()
        }

        fragmentCameraBinding.button3.setOnClickListener {
            if (fragmentCameraBinding.button3.text.toString() == "Recognize") {
                start = true
                fragmentCameraBinding.textAbovePreview.text = "Recognized Face:"
                fragmentCameraBinding.button3.text = "Add Face"
                fragmentCameraBinding.imageButton.visibility = View.INVISIBLE
                fragmentCameraBinding.textView.visibility = View.VISIBLE
                fragmentCameraBinding.imageView.visibility = View.INVISIBLE
                fragmentCameraBinding.textView2.text = ""
            } else {
                fragmentCameraBinding.textAbovePreview.text = "Face Preview: "
                fragmentCameraBinding.button3.text = "Recognize"
                fragmentCameraBinding.imageButton.visibility = View.VISIBLE
                fragmentCameraBinding.textView.visibility = View.INVISIBLE
                fragmentCameraBinding.imageView.visibility = View.VISIBLE
                fragmentCameraBinding.textView2.text =
                    "1.Bring Face in view of Camera.\n\n2.Your Face preview will appear here.\n\n3.Click Add button to save face."
            }
        }

        fragmentCameraBinding.imageButton.setOnClickListener {
            addFace()
        }
    }

    private fun addFace() {
        start = false
        val builder = AlertDialog.Builder(requireContext())
        builder.setTitle("Enter Name")

        val input = EditText(requireContext())
        input.inputType = InputType.TYPE_CLASS_TEXT
        builder.setView(input)

        builder.setPositiveButton("ADD") { _, _ ->
            val result = SimilarityClassifier.Recognition("0", "", -1f)
            result.extra = embeddings

            registered[input.text.toString()] = result
            insertToSP(registered, 0) // Save faces immediately after adding
            start = true
        }
        builder.setNegativeButton("Cancel") { dialog, _ ->
            start = true
            dialog.cancel()
        }

        builder.show()
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
        cameraSelector = CameraSelector.Builder().requireLensFacing(camFace).build()
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
                            start -> analyzeFace(image)
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

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun analyzeFace(image: ImageProxy) {
        val mediaImage = image.image
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, image.imageInfo.rotationDegrees)
            detector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        val face = faces[0]
                        val frameBmp = toBitmap(mediaImage)
                        val rot = image.imageInfo.rotationDegrees
                        val frameBmp1 = rotateBitmap(frameBmp, rot, false, false)
                        val boundingBox = RectF(face.boundingBox)
                        val croppedFace = getCropBitmapByCPU(frameBmp1, boundingBox)
                        var scaled = getResizedBitmap(croppedFace, 112, 112)
                        if (flipX) {
                            scaled = rotateBitmap(scaled, 0, flipX, false)
                        }
                        if (start) recognizeImage(scaled)
                    } else {
                        fragmentCameraBinding.textView.text =
                            if (registered.isEmpty()) "Add Face" else "No Face Detected!"
                    }
                    image.close()
                }
                .addOnFailureListener {
                    Log.e(TAG, "Face detection failed: ${it.message}")
                    image.close()
                }
        } else {
            image.close()
        }
    }

    private fun recognizeImage(bitmap: Bitmap) {
        fragmentCameraBinding.imageView.setImageBitmap(bitmap)

        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())

        intValues = IntArray(inputSize * inputSize)

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        imgData.rewind()

        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) {
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else {
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - imageMean) / imageStd)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - imageMean) / imageStd)
                    imgData.putFloat(((pixelValue and 0xFF) - imageMean) / imageStd)
                }
            }
        }

        val inputArray = arrayOf<Any>(imgData)

        val outputMap = HashMap<Int, Any>()

        embeddings = Array(1) { FloatArray(outputSize) }

        outputMap[0] = embeddings

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)

        var distanceLocal = Float.MAX_VALUE
        var id = "0"
        var label = "?"

        if (registered.isNotEmpty()) {
            val nearest = findNearest(embeddings[0])

            if (nearest[0] != null) {
                val name = nearest[0].first
                distanceLocal = nearest[0].second
                fragmentCameraBinding.textView.text = if (developerMode) {
                    if (distanceLocal < distance) {
                        "Nearest: $name\nDist: %.3f".format(distanceLocal) +
                                "\n2nd Nearest: ${nearest[1].first}\nDist: %.3f".format(nearest[1].second)
                    } else {
                        "Unknown\nDist: %.3f".format(distanceLocal) +
                                "\nNearest: $name\nDist: %.3f".format(distanceLocal) +
                                "\n2nd Nearest: ${nearest[1].first}\nDist: %.3f".format(nearest[1].second)
                    }
                } else {
                    if (distanceLocal < distance) {
                        name
                    } else {
                        "Unknown"
                    }
                }
            }
        }
    }


    private fun findNearest(emb: FloatArray): List<Pair<String, Float>> {
        val neighbourList = mutableListOf<Pair<String, Float>>()
        var ret: Pair<String, Float>? = null
        var prevRet: Pair<String, Float>? = null

        registered.forEach { (name, value) ->
            val knownEmb = (value.extra as Array<FloatArray>)[0]

            var distance = 0f
            for (i in emb.indices) {
                val diff = emb[i] - knownEmb[i]
                distance += diff * diff
            }
            distance = Math.sqrt(distance.toDouble()).toFloat()

            if (ret == null || distance < ret!!.second) {
                prevRet = ret
                ret = Pair(name, distance)
            }
        }

        if (prevRet == null) prevRet = ret

        neighbourList.add(ret!!)
        neighbourList.add(prevRet!!)

        return neighbourList
    }

    private fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        val width = bm.width
        val height = bm.height
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        val resizedBitmap = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false)
        bm.recycle()
        return resizedBitmap
    }

    private fun getCropBitmapByCPU(source: Bitmap, cropRectF: RectF): Bitmap {
        val resultBitmap = Bitmap.createBitmap(cropRectF.width().toInt(), cropRectF.height().toInt(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        paint.color = Color.WHITE
        canvas.drawRect(RectF(0f, 0f, cropRectF.width(), cropRectF.height()), paint)
        val matrix = Matrix()
        matrix.postTranslate(-cropRectF.left, -cropRectF.top)
        canvas.drawBitmap(source, matrix, paint)
        if (!source.isRecycled) {
            source.recycle()
        }
        return resultBitmap
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int, flipX: Boolean, flipY: Boolean): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        matrix.postScale(if (flipX) -1.0f else 1.0f, if (flipY) -1.0f else 1.0f)
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotatedBitmap != bitmap) {
            bitmap.recycle()
        }
        return rotatedBitmap
    }

    private fun YUV_420_888toNV21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4

        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val rowStride = image.planes[0].rowStride

        var pos = 0

        if (rowStride == width) {
            yBuffer.get(nv21, 0, ySize)
            pos += ySize
        } else {
            var yBufferPos = -rowStride
            while (pos < ySize) {
                yBufferPos += rowStride
                yBuffer.position(yBufferPos)
                yBuffer.get(nv21, pos, width)
                pos += width
            }
        }

        val uvRowStride = image.planes[2].rowStride
        val pixelStride = image.planes[2].pixelStride

        if (pixelStride == 2 && uvRowStride == width && uBuffer[0] == vBuffer[1]) {
            val savePixel = vBuffer[1]
            vBuffer.put(1, (savePixel.toInt() xor -1).toByte())
            if (uBuffer[0] == (savePixel.toInt() xor -1).toByte()) {
                vBuffer.put(1, savePixel)
                vBuffer.position(0)
                uBuffer.position(0)
                vBuffer.get(nv21, ySize, 1)
                uBuffer.get(nv21, ySize + 1, uBuffer.remaining())
                return nv21
            }
            vBuffer.put(1, savePixel)
        }

        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val vuPos = col * pixelStride + row * uvRowStride
                nv21[pos++] = vBuffer[vuPos]
                nv21[pos++] = uBuffer[vuPos]
            }
        }

        return nv21
    }

    private fun toBitmap(image: Image): Bitmap {
        val nv21 = YUV_420_888toNV21(image)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)

        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun insertToSP(jsonMap: HashMap<String, SimilarityClassifier.Recognition>, mode: Int) {
        if (mode == 1) jsonMap.clear()
        else if (mode == 0) jsonMap.putAll(readFromSP())

        val jsonString = Gson().toJson(jsonMap)
        val sharedPreferences = requireContext().getSharedPreferences("HashMap", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putString("map", jsonString)
        editor.apply()
        Toast.makeText(requireContext(), "Recognitions Saved", Toast.LENGTH_SHORT).show()
    }

    private fun readFromSP(): HashMap<String, SimilarityClassifier.Recognition> {
        val sharedPreferences = requireContext().getSharedPreferences("HashMap", Context.MODE_PRIVATE)
        val defValue = Gson().toJson(HashMap<String, SimilarityClassifier.Recognition>())
        val json = sharedPreferences.getString("map", defValue)
        val token = object : TypeToken<HashMap<String, SimilarityClassifier.Recognition>>() {}
        val retrievedMap: HashMap<String, SimilarityClassifier.Recognition> = Gson().fromJson(json, token.type)

        retrievedMap.forEach { (key, value) ->
            val output = Array(1) { FloatArray(outputSize) }
            val arrayList = value.extra as ArrayList<*>
            val floatArray = arrayList[0] as ArrayList<*>
            for (counter in floatArray.indices) {
                output[0][counter] = (floatArray[counter] as Double).toFloat()
            }
            value.extra = output
        }

        Toast.makeText(requireContext(), "Recognitions Loaded", Toast.LENGTH_SHORT).show()
        return retrievedMap
    }

    private fun loadPhoto() {
        start = false
        val intent = Intent().apply {
            type = "image/*"
            action = Intent.ACTION_GET_CONTENT
        }
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), selectPicture)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == selectPicture) {
                val selectedImageUri = data?.data
                selectedImageUri?.let {
                    val impPhoto = InputImage.fromBitmap(getBitmapFromUri(it), 0)
                    detector.process(impPhoto)
                        .addOnSuccessListener { faces ->
                            if (faces.isNotEmpty()) {
                                fragmentCameraBinding.button3.text = "Recognize"
                                fragmentCameraBinding.imageButton.visibility = View.VISIBLE
                                fragmentCameraBinding.textView.visibility = View.INVISIBLE
                                fragmentCameraBinding.imageView.visibility = View.VISIBLE
                                fragmentCameraBinding.textView2.text =
                                    "1.Bring Face in view of Camera.\n\n2.Your Face preview will appear here.\n\n3.Click Add button to save face."
                                val face = faces[0]
                                var frameBmp = getBitmapFromUri(it)
                                frameBmp = rotateBitmap(frameBmp, 0, flipX, false)
                                val boundingBox = RectF(face.boundingBox)
                                val croppedFace = getCropBitmapByCPU(frameBmp, boundingBox)
                                val scaled = getResizedBitmap(croppedFace, 112, 112)
                                recognizeImage(scaled)
                                addFace()
                            }
                        }
                        .addOnFailureListener {
                            start = true
                            Toast.makeText(requireContext(), "Failed to add", Toast.LENGTH_SHORT).show()
                        }
                    fragmentCameraBinding.imageView.setImageBitmap(getBitmapFromUri(it))
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun getBitmapFromUri(uri: Uri): Bitmap {
        val parcelFileDescriptor = requireContext().contentResolver.openFileDescriptor(uri, "r")
        val fileDescriptor: FileDescriptor = parcelFileDescriptor!!.fileDescriptor
        val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
        parcelFileDescriptor.close()
        return image
    }

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, modelFile: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
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

                    val objectRealHeight = 0.2f
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