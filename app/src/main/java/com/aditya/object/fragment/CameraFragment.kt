package com.aditya.`object`.fragment

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.*
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
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.core.Camera
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.*
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
    private lateinit var actions: Button

    private val RECORD_REQUEST_CODE = 101
    private val RESTART_DELAY_MS = 1000L
    private val REQUEST_LOCATION_PERMISSION = 102

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var textToSpeech: TextToSpeech
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var previousDetectedObject: String? = null

    private var isWaitingForDestination = false
    private var currentLatLng: LatLng? = null

    private lateinit var detector: FaceDetector
    private lateinit var tfLite: Interpreter
    private var registered = HashMap<String, SimilarityClassifier.Recognition>()
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

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(KEY_IS_OBJECT_DETECTION, isObjectDetection)
        outState.putBoolean(KEY_IS_TEXT_RECOGNITION, isTextRecognition)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.Builder().build())
        checkPermission()
        return fragmentCameraBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        actions = view.findViewById(R.id.button2)  // Initialize actions button here

        initButtons()

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
            "switch camera" -> {
                switchCamera()
            }
            "add face" -> {
                addFaceVoiceCommand()
            }
            "read face" -> {
                startFaceRecognition()
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
            switchCamera()
        }

        actions.setOnClickListener {
            val builder = context?.let { it1 -> androidx.appcompat.app.AlertDialog.Builder(it1) }
            if (builder != null) {
                builder.setTitle("Select Action:")
            }

            val names = arrayOf(
                "View Recognition List",
                "Update Recognition List",
                "Save Recognitions",
                "Load Recognitions",
                "Clear All Recognitions",
                "Import Photo (Beta)",
                "Hyperparameters",
                "Developer Mode"
            )

            if (builder != null) {
                builder.setItems(names) { _, which ->
                    when (which) {
                        0 -> displayNameListView()
                        1 -> updateNameListView()
                        2 -> insertToSP(registered, 0) // mode: 0:save all, 1:clear all, 2:update all
                        3 -> registered.putAll(readFromSP())
                        4 -> clearNameList()
                        5 -> loadPhoto()
                        6 -> testHyperparameter()
                        7 -> developerMode()
                    }
                }
            }

            if (builder != null) {
                builder.setPositiveButton("OK", null)
            }
            if (builder != null) {
                builder.setNegativeButton("Cancel", null)
            }

            val dialog = builder?.create()
            if (dialog != null) {
                dialog.show()
            }
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

    private fun testHyperparameter() {
        val builder = context?.let { androidx.appcompat.app.AlertDialog.Builder(it) }
        if (builder != null) {
            builder.setTitle("Select Hyperparameter:")
        }

        val names = arrayOf("Maximum Nearest Neighbour Distance")

        if (builder != null) {
            builder.setItems(names) { _, which ->
                if (which == 0) {
                    hyperparameters()
                }
            }
        }

        if (builder != null) {
            builder.setPositiveButton("OK", null)
        }
        if (builder != null) {
            builder.setNegativeButton("Cancel", null)
        }

        val dialog = builder?.create()
        if (dialog != null) {
            dialog.show()
        }
    }

    private fun developerMode() {
        developerMode = if (developerMode) {
            Toast.makeText(context, "Developer Mode OFF", Toast.LENGTH_SHORT).show()
            false
        } else {
            Toast.makeText(context, "Developer Mode ON", Toast.LENGTH_SHORT).show()
            true
        }
    }


    private fun clearNameList() {
        val builder = context?.let { androidx.appcompat.app.AlertDialog.Builder(it) }
        if (builder != null) {
            builder.setTitle("Do you want to delete all Recognitions?")
        }
        if (builder != null) {
            builder.setPositiveButton("Delete All") { _, _ ->
                registered.clear()
                Toast.makeText(context, "Recognitions Cleared", Toast.LENGTH_SHORT).show()
            }
        }
        insertToSP(registered, 1)
        if (builder != null) {
            builder.setNegativeButton("Cancel", null)
        }
        val dialog = builder?.create()
        if (dialog != null) {
            dialog.show()
        }
    }

    private fun updateNameListView() {
        val builder = context?.let { androidx.appcompat.app.AlertDialog.Builder(it) }
        if (registered.isEmpty()) {
            builder?.setTitle("No Faces Added!!")
            builder?.setPositiveButton("OK", null)
        } else {
            builder?.setTitle("Select Recognition to delete:")

            // Create a list of names with phone numbers
            val namesWithNumbers = Array(registered.size) { i ->
                val recognition = registered.values.elementAt(i)
                val name = registered.keys.elementAt(i)
                val phoneNumber = recognition.phoneNumber ?: "N/A" // Use "N/A" if phoneNumber is null
                "$name ($phoneNumber)"
            }

            val checkedItems = BooleanArray(registered.size)

            builder?.setMultiChoiceItems(namesWithNumbers, checkedItems) { _, which, isChecked ->
                checkedItems[which] = isChecked
            }

            builder?.setPositiveButton("OK") { _, _ ->
                for (i in checkedItems.indices) {
                    if (checkedItems[i]) {
                        registered.remove(registered.keys.elementAt(i))
                    }
                }
                insertToSP(registered, 2) // mode: 0:save all, 1:clear all, 2:update all
                Toast.makeText(context, "Recognitions Updated", Toast.LENGTH_SHORT).show()
            }

            builder?.setNegativeButton("Cancel", null)
        }

        val dialog = builder?.create()
        dialog?.show()
    }

    private fun displayNameListView() {
        val builder = context?.let { androidx.appcompat.app.AlertDialog.Builder(it) }
        if (builder != null) {
            builder.setTitle(if (registered.isEmpty()) "No Faces Added!!" else "Recognitions:")
        }

        val names = Array(registered.size) { i -> registered.keys.elementAt(i) }
        if (builder != null) {
            builder.setItems(names, null)
        }

        if (builder != null) {
            builder.setPositiveButton("OK", null)
        }

        val dialog = builder?.create()
        if (dialog != null) {
            dialog.show()
        }
    }

    private fun hyperparameters() {
        val builder = context?.let { androidx.appcompat.app.AlertDialog.Builder(it) }
        if (builder != null) {
            builder.setTitle("Euclidean Distance")
        }
        if (builder != null) {
            builder.setMessage("0.00 -> Perfect Match\n1.00 -> Default\nTurn On Developer Mode to find optimum value\n\nCurrent Value:")
        }

        val input = EditText(context)
        input.inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        if (builder != null) {
            builder.setView(input)
        }

        val sharedPref = requireActivity().getSharedPreferences("Distance", Context.MODE_PRIVATE)
        distance = sharedPref.getFloat("distance", 1.00f)
        input.setText(distance.toString())

        if (builder != null) {
            builder.setPositiveButton("Update") { _, _ ->
                distance = input.text.toString().toFloat()

                val editor = sharedPref.edit()
                editor.putFloat("distance", distance)
                editor.apply()
            }
        }
        if (builder != null) {
            builder.setNegativeButton("Cancel") { dialog, _ ->
                dialog.cancel()
            }
        }

        if (builder != null) {
            builder.show()
        }
    }

    private fun initializeFaceRecognition() {
        registered = readFromSP()
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

    private fun addFaceVoiceCommand() {
        fragmentCameraBinding.button3.performClick()
        fragmentCameraBinding.imageButton.performClick()

        textToSpeech.speak("Please say the name of the person after the beep", TextToSpeech.QUEUE_FLUSH, null, null)

        setupSpeechRecognizerForName()
    }

    private fun setupSpeechRecognizerForName() {
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}

            override fun onBeginningOfSpeech() {}

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                Log.e(TAG, "Speech Recognition Error: $error")
            }

            override fun onResults(results: Bundle?) {
                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.let { matches ->
                    if (matches.isNotEmpty()) {
                        val name = matches[0]
                        // Introduce a 30 seconds delay before saving the face
                        lifecycleScope.launch {
                            delay(30000)  // 30 seconds delay
                            promptForPhoneNumber(name)
                        }
                    }
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        speechRecognizer.startListening(speechRecognizerIntent)
    }

    private fun promptForPhoneNumber(name: String) {
        val builder = AlertDialog.Builder(requireContext())
        builder.setTitle("Enter Phone Number for $name")

        val input = EditText(requireContext())
        input.inputType = InputType.TYPE_CLASS_PHONE
        builder.setView(input)

        builder.setPositiveButton("Add") { _, _ ->
            val phoneNumber = input.text.toString()
            saveFace(name, phoneNumber)
        }

        builder.setNegativeButton("Cancel") { dialog, _ ->
            dialog.cancel()
        }

        builder.show()
    }

    private fun saveFace(name: String, phoneNumber: String) {
        val result = SimilarityClassifier.Recognition("0", name, -1f)
        result.extra = embeddings
        result.phoneNumber = phoneNumber  // Set the phone number here
        registered[name] = result
        insertToSP(registered, 0)
        start = true
        Toast.makeText(requireContext(), "$name with Phone Number $phoneNumber has been added", Toast.LENGTH_SHORT).show()
    }

    private fun startFaceRecognition() {
        start = true
        Toast.makeText(requireContext(), "Face Recognition Started", Toast.LENGTH_SHORT).show()
    }

    private fun switchCamera(){
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

    private fun addFace() {
        start = false
        val builder = AlertDialog.Builder(requireContext())
        builder.setTitle("Enter Name")

        val nameInput = EditText(requireContext())
        nameInput.inputType = InputType.TYPE_CLASS_TEXT
        builder.setView(nameInput)

        builder.setPositiveButton("NEXT") { _, _ ->
            val name = nameInput.text.toString()
            promptForPhoneNumber(name)
        }

        builder.setNegativeButton("Cancel") { dialog, _ ->
            start = true
            dialog.cancel()
        }

        builder.show()
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

        if (registered.isNotEmpty()) {
            val nearest = findNearest(embeddings[0])

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
        if (resultCode == Activity.RESULT_OK) {
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
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
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
