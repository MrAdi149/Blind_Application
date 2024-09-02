package com.aditya.`object`.fragment

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.app.AlertDialog
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.YuvImage
import android.hardware.usb.UsbDevice
import android.media.Image
import android.os.*
import android.provider.MediaStore
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.PopupWindow
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.view.children
import androidx.core.widget.TextViewCompat
import androidx.lifecycle.lifecycleScope
import com.aditya.`object`.DetectionResult
import com.aditya.`object`.ObjectDetectorHelper
import com.aditya.`object`.R
import com.aditya.`object`.SimilarityClassifier
import com.aditya.`object`.databinding.FragmentUsbBinding
import com.afollestad.materialdialogs.MaterialDialog
import com.afollestad.materialdialogs.list.listItemsSingleChoice
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.jiangdg.ausbc.MultiCameraClient
import com.jiangdg.ausbc.base.CameraFragment
import com.jiangdg.ausbc.callback.ICameraStateCallBack
import com.jiangdg.ausbc.callback.ICaptureCallBack
import com.jiangdg.ausbc.callback.IPlayCallBack
import com.jiangdg.ausbc.callback.IPreviewDataCallBack
import com.jiangdg.ausbc.camera.CameraUVC
import com.jiangdg.ausbc.render.effect.EffectBlackWhite
import com.jiangdg.ausbc.render.effect.EffectSoul
import com.jiangdg.ausbc.render.effect.EffectZoom
import com.jiangdg.ausbc.render.effect.bean.CameraEffect
import com.jiangdg.ausbc.utils.*
import com.jiangdg.ausbc.utils.bus.BusKey
import com.jiangdg.ausbc.utils.bus.EventBus
import com.jiangdg.ausbc.widget.*
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Executors

class UsbFragment : CameraFragment(), View.OnClickListener, CaptureMediaView.OnViewClickListener {
    private var mMoreMenu: PopupWindow? = null
    private var isCapturingVideoOrAudio: Boolean = false
    private var isPlayingMic: Boolean = false
    private var mRecTimer: Timer? = null
    private var mRecSeconds = 0
    private var mRecMinute = 0
    private var mRecHours = 0
    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var textRecognizer: TextRecognizer
    private var isObjectDetectionActive: Boolean = false
    private var isTextRecognitionActive: Boolean = false
    private var isFaceRecognitionActive: Boolean = false
    private val inferenceDispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val speechQueue: Queue<String> = LinkedList()
    private var lastSpokenTime: Long = 0
    private val MIN_SPEECH_INTERVAL_MS = 2000L
    private val SPEECH_RECOGNITION_DELAY = 1000L
    private var previewDataCallback: IPreviewDataCallBack? = null
    private var lastFrameTime: Long = 0L

    private lateinit var tfLite: Interpreter
    private var registered = HashMap<String, SimilarityClassifier.Recognition>()
    private var start = true
    private var flipX = false
    private var isModelQuantized = false
    private val inputSize = 112
    private lateinit var embeddings: Array<FloatArray>
    private lateinit var intValues: IntArray
    private val imageMean = 128.0f
    private val imageStd = 128.0f
    private val outputSize = 192
    private lateinit var faceDetector: FaceDetector
    private val modelFile = "mobile_face_net.tflite"
    private var developerMode = false
    private var distance = 1.0f
    private val selectPicture = 1

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private var isListening: Boolean = false

    private var previousDetections: MutableSet<String> = mutableSetOf()
    private var detectedObjects: MutableMap<String, Float> = mutableMapOf()

    private lateinit var recognizedFaceImageView: ImageView

    private val mCameraModeTabMap = mapOf(
        CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC to R.id.takePictureModeTv,
        CaptureMediaView.CaptureMode.MODE_CAPTURE_VIDEO to R.id.recordVideoModeTv,
        CaptureMediaView.CaptureMode.MODE_CAPTURE_AUDIO to R.id.recordAudioModeTv
    )

    private fun initializeEmbeddings() {
        embeddings = Array(1) { FloatArray(outputSize) }
    }

    private val ttsListener = TextToSpeech.OnUtteranceCompletedListener {
        if (speechQueue.isNotEmpty()) {
            val nextText = speechQueue.poll()
            textToSpeech.speak(nextText, TextToSpeech.QUEUE_FLUSH, null, nextText.hashCode().toString())
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
                ToastUtils.showToast(requireContext(), "Listening for voice command...")
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

    private fun startListeningForVoiceCommand() {
        try {
            speechRecognizer.startListening(speechRecognizerIntent)
        } catch (e: ActivityNotFoundException) {
            Log.e(TAG, "Speech recognition not supported on this device: ${e.message}")
            Toast.makeText(requireContext(), "Speech recognition not supported on this device", Toast.LENGTH_SHORT).show()
        }
    }

    private fun restartSpeechRecognizer() {
        lifecycleScope.launch {
            delay(SPEECH_RECOGNITION_DELAY)
            if (!isListening) {
                startListeningForVoiceCommand()
            }
        }
    }

    private fun handleVoiceCommand(command: String) {
        val words = command.split(" ")

        if (words.size == 2 && words[1] == "distance") {
            val objectName = words[0]
            val distance = detectedObjects[objectName]
            if (distance != null) {
                speakDistance(objectName, distance)
            } else {
                Log.d(TAG, "Distance for $objectName not available.")
                textToSpeech.speak("Distance for $objectName not available", TextToSpeech.QUEUE_FLUSH, null, objectName.hashCode().toString())
            }
            return
        }

        when (command.toLowerCase(Locale.ROOT)) {
            "object" -> {
                if (!isObjectDetectionActive) {
                    stopTextDetection()
                    startObjectDetection()
                }
            }
            "read" -> {
                if (!isTextRecognitionActive) {
                    stopObjectDetection()
                    startTextDetection()
                }
            }
            "stop" -> {
                stopAllDetections()
                stopObjectDetection()
            }
            else -> {
                Log.d(TAG, "Unknown command: $command")
            }
        }
    }

    private fun speakDistance(objectName: String, distance: Float) {
        val message = String.format(Locale.US, "The distance to %s is approximately %.2f meters", objectName, distance)
        Log.d(TAG, message)
        textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, message.hashCode().toString())
    }

    private val mEffectDataList by lazy {
        arrayListOf(
            CameraEffect.NONE_FILTER,
            CameraEffect(
                EffectBlackWhite.ID,
                "BlackWhite",
                CameraEffect.CLASSIFY_ID_FILTER,
                effect = EffectBlackWhite(requireActivity()),
                coverResId = R.mipmap.filter0
            ),
            CameraEffect.NONE_ANIMATION,
            CameraEffect(
                EffectZoom.ID,
                "Zoom",
                CameraEffect.CLASSIFY_ID_ANIMATION,
                effect = EffectZoom(requireActivity()),
                coverResId = R.mipmap.filter2
            ),
            CameraEffect(
                EffectSoul.ID,
                "Soul",
                CameraEffect.CLASSIFY_ID_ANIMATION,
                effect = EffectSoul(requireActivity()),
                coverResId = R.mipmap.filter1
            ),
        )
    }

    private lateinit var mViewBinding: FragmentUsbBinding
    private var mCameraMode = CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC

    private val mMainHandler: Handler by lazy {
        Handler(Looper.getMainLooper()) {
            when (it.what) {
                WHAT_START_TIMER -> {
                    if (mRecSeconds % 2 != 0) {
                        mViewBinding.recStateIv.visibility = View.VISIBLE
                    } else {
                        mViewBinding.recStateIv.visibility = View.INVISIBLE
                    }
                    mViewBinding.recTimeTv.text = calculateTime(mRecSeconds, mRecMinute)
                }
                WHAT_STOP_TIMER -> {
                    mViewBinding.modeSwitchLayout.visibility = View.VISIBLE
                    mViewBinding.toolbarGroup.visibility = View.VISIBLE
                    mViewBinding.albumPreviewIv.visibility = View.VISIBLE
                    mViewBinding.lensFacingBtn1.visibility = View.VISIBLE
                    mViewBinding.recTimerLayout.visibility = View.GONE
                    mViewBinding.recTimeTv.text = calculateTime(0, 0)
                }
            }
            true
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.US
                textToSpeech.setSpeechRate(0.8f)
                textToSpeech.setPitch(1.0f)
                textToSpeech.setOnUtteranceCompletedListener(ttsListener)
            }
        }

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.Builder().build())

        recognizedFaceImageView = view.findViewById(R.id.imageView)

        initObjectDetector()
        initFaceRecognition()

        setupSpeechRecognizer()
        startListeningForVoiceCommand()
    }

    private fun shouldSpeakNow(): Boolean {
        val currentTime = System.currentTimeMillis()
        return if (currentTime - lastSpokenTime >= MIN_SPEECH_INTERVAL_MS) {
            lastSpokenTime = currentTime
            true
        } else {
            false
        }
    }

    private fun startObjectDetection() {
        if (!isObjectDetectionActive) {
            isObjectDetectionActive = true
            Toast.makeText(requireContext(), "Object Detection Started", Toast.LENGTH_SHORT).show()

            previewDataCallback = object : IPreviewDataCallBack {
                override fun onPreviewData(
                    data: ByteArray?,
                    width: Int,
                    height: Int,
                    format: IPreviewDataCallBack.DataFormat
                ) {
                    Log.d(TAG, "Preview Data Received: Width = $width, Height = $height, Format = $format")
                    processFrame(data, width, height, format)
                }
            }

            getCurrentCamera()?.addPreviewDataCallBack(previewDataCallback!!)
        }
    }

    private fun processFrame(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {
        data ?: return

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFrameTime < DETECTION_INTERVAL_MS) return
        lastFrameTime = currentTime

        Log.d(TAG, "Processing frame with width: $width, height: $height, format: $format")

        lifecycleScope.launch(inferenceDispatcher) {
            try {
                val bitmap = convertYUVToBitmap(data, width, height, format)
                objectDetectorHelper.detect(bitmap, 0)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}")
            }
        }
    }

    private fun convertYUVToBitmap(data: ByteArray, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat): Bitmap {
        return if (format == IPreviewDataCallBack.DataFormat.RGBA) {
            val pixels = IntArray(width * height)
            for (i in 0 until width * height) {
                val r = data[i * 4].toInt() and 0xFF
                val g = data[i * 4 + 1].toInt() and 0xFF
                val b = data[i * 4 + 2].toInt() and 0xFF
                val a = data[i * 4 + 3].toInt() and 0xFF
                pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }

            Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        } else {
            val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }
    }

    private fun initFaceRecognition() {
        registered = HashMap()
        tfLite = Interpreter(loadModelFile(requireContext(), modelFile))

        val highAccuracyOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
        faceDetector = FaceDetection.getClient(highAccuracyOpts)

        val button3: Button = mViewBinding.button3
        val imageButton: ImageButton = mViewBinding.imageButton
        val textView2: TextView = mViewBinding.textView2
        val textAbovePreview: TextView = mViewBinding.textAbovePreview
        val recognizedFaceImageView: ImageView = mViewBinding.imageView

        button3.text = "Add Face"
        imageButton.visibility = View.GONE
        recognizedFaceImageView.visibility = View.GONE
        textView2.visibility = View.GONE

        button3.setOnClickListener {
            if (button3.text == "Add Face") {
                button3.text = "Recognize"
                imageButton.visibility = View.VISIBLE
                recognizedFaceImageView.visibility = View.VISIBLE
                textView2.visibility = View.VISIBLE
                textAbovePreview.text = "Face Preview: "
                textView2.text = "1. Bring Face in view of Camera.\n\n2. Your Face preview will appear here.\n\n3. Click Add button to save face."
                startFaceRecognition()
            } else {
                button3.text = "Add Face"
                imageButton.visibility = View.GONE
                recognizedFaceImageView.visibility = View.GONE
                textView2.visibility = View.GONE
                textAbovePreview.text = "Recognized Face:"
                textView2.text = ""
                stopFaceRecognition()
            }
        }

        val actionsButton: Button = mViewBinding.button2
        actionsButton.setOnClickListener {
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
                        2 -> insertToSP(registered, 0)
                        3 -> registered.putAll(readFromSP())
                        4 -> clearNameList()
                        5 -> loadPhoto()
                        6 -> testHyperparameter()
                        7 -> toggleDeveloperMode()
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
            dialog?.show()
        }

        mViewBinding.imageButton.setOnClickListener {
            addFace()
        }
    }

    private fun toggleDeveloperMode() {
        developerMode = !developerMode
        Toast.makeText(requireContext(), "Developer Mode: ${if (developerMode) "ON" else "OFF"}", Toast.LENGTH_SHORT).show()
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

            val namesWithNumbers = Array(registered.size) { i ->
                val recognition = registered.values.elementAt(i)
                val name = registered.keys.elementAt(i)
                val phoneNumber = recognition.phoneNumber ?: "N/A"
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
                insertToSP(registered, 2)
                Toast.makeText(context, "Recognitions Updated", Toast.LENGTH_SHORT).show()
            }

            builder?.setNegativeButton("Cancel", null)
        }

        val dialog = builder?.create()
        dialog?.show()
    }

    private fun loadPhoto() {
        start = false
        val intent = Intent().apply {
            type = "image/*"
            action = Intent.ACTION_GET_CONTENT
        }
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), selectPicture)
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

    @Throws(IOException::class)
    private fun loadModelFile(activity: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun startFaceRecognition() {
        if (!isFaceRecognitionActive) {
            isFaceRecognitionActive = true
            initializeEmbeddings()
            Toast.makeText(requireContext(), "Face Recognition Started", Toast.LENGTH_SHORT).show()

            previewDataCallback = object : IPreviewDataCallBack {
                override fun onPreviewData(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {
                    data?.let {
                        processFaceRecognition(it, width, height, format)
                    }
                }
            }
            getCurrentCamera()?.addPreviewDataCallBack(previewDataCallback!!)
        }
    }

    private fun stopFaceRecognition() {
        isFaceRecognitionActive = false
        previewDataCallback?.let {
            getCurrentCamera()?.removePreviewDataCallBack(it)
            previewDataCallback = null
        }
        Toast.makeText(requireContext(), "Face Recognition Stopped", Toast.LENGTH_SHORT).show()
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

        if (!::embeddings.isInitialized) {
            initializeEmbeddings()
        }

        val result = SimilarityClassifier.Recognition("0", name, -1f)
        result.extra = embeddings
        result.phoneNumber = phoneNumber
        registered[name] = result
        insertToSP(registered, 0)
        start = true
        Toast.makeText(requireContext(), "$name with Phone Number $phoneNumber has been added", Toast.LENGTH_SHORT).show()
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

    private fun processFaceRecognition(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {
        data ?: return

        val currentTime = System.currentTimeMillis()
        val frameRate = if (lastFrameTime > 0) 1000 / (currentTime - lastFrameTime).toFloat() else 0f
        if (currentTime - lastFrameTime < DETECTION_INTERVAL_MS) return
        lastFrameTime = currentTime

        Log.d(TAG, "Current Frame Rate: $frameRate FPS")

        lifecycleScope.launch(inferenceDispatcher) {
            try {
                val bitmap = convertYUVToBitmap(data, width , height , format)

                analyzeFace(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}")
            }
        }
    }

    private fun analyzeFace(bitmap: Bitmap) {
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    Log.d(TAG, "Face detected: ${faces.size}")
                    val face = faces[0]
                    val boundingBox = RectF(face.boundingBox)
                    val croppedFace = getCropBitmapByCPU(bitmap, boundingBox)
                    var scaled = getResizedBitmap(croppedFace, inputSize, inputSize)
                    if (flipX) {
                        scaled = rotateBitmap(scaled, 0, flipX, false)
                    }
                    if (start) {
                        Log.d(TAG, "Starting recognition for cropped face.")
                        recognizeImage(scaled)
                    }
                } else {
                    Log.d(TAG, "No face detected in the current frame.")
                    mViewBinding.textView.text = if (registered.isEmpty()) "Add Face" else "No Face Detected!"
                }
            }
            .addOnFailureListener {
                Log.e(TAG, "Face detection failed: ${it.message}")
            }
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

    private fun recognizeImage(bitmap: Bitmap) {
        mViewBinding.imageView.setImageBitmap(bitmap)

        Log.d(TAG, "Running recognition on bitmap of size: ${bitmap.width}x${bitmap.height}")

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        mViewBinding.imageView.setImageBitmap(resizedBitmap)

        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())

        intValues = IntArray(inputSize * inputSize)

        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

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
            Log.d(TAG, "Recognition result: $name with distance $distanceLocal")
            mViewBinding.textView.text = if (developerMode) {
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

    private fun initObjectDetector() {
        lifecycleScope.launch(inferenceDispatcher) {
            objectDetectorHelper = ObjectDetectorHelper(
                context = requireContext(),
                objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                    override fun onError(error: String) {
                        Log.e(TAG, "Object Detection Error: $error")
                        ToastUtils.show("Object Detection Error: $error")
                    }

                    override fun onResults(
                        results: MutableList<Detection>?,
                        inferenceTime: Long,
                        imageHeight: Int,
                        imageWidth: Int
                    ) {
                        results?.let { detections ->
                            if (detections.isNotEmpty()) {
                                Log.d(TAG, "Detections found: ${detections.size}")
                                val detectionsWithDistance = detections.map { detection ->
                                    val distance = objectDetectorHelper.estimateDistance(detection.boundingBox, 0.2f)
                                    DetectionResult(detection, distance)
                                }
                                updateOverlayView(detectionsWithDistance, imageHeight, imageWidth)
                                handleDetections(detectionsWithDistance)
                            } else {
                                Log.d(TAG, "No detections found.")
                            }
                        }
                    }
                }
            )
        }
    }

    private fun handleDetections(detections: List<DetectionResult>) {
        val currentDetections = mutableSetOf<String>()
        detectedObjects.clear()

        for (result in detections) {
            val detection = result.detection
            detection.categories.firstOrNull()?.label?.let { label ->
                currentDetections.add(label)
                detectedObjects[label] = result.distance
            }
        }

        val newDetections = currentDetections.subtract(previousDetections)
        for (detection in newDetections) {
            if (shouldSpeakNow()) {
                Log.d(TAG, "Detected new object: $detection")
                if (!textToSpeech.isSpeaking) {
                    textToSpeech.speak(detection, TextToSpeech.QUEUE_FLUSH, null, detection.hashCode().toString())
                } else {
                    speechQueue.offer(detection)
                }
            }
        }
        previousDetections = currentDetections
    }

    private fun updateOverlayView(detections: List<DetectionResult>, imageHeight: Int, imageWidth: Int) {
        mViewBinding.overlay.setResults(detections, imageHeight, imageWidth)
    }

    private fun stopLiveObjectDetection() {
        isObjectDetectionActive = false
        getCurrentCamera()?.removePreviewDataCallBack(object : IPreviewDataCallBack {
            override fun onPreviewData(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {}
        })
    }

    override fun initView() {
        super.initView()
        mViewBinding.lensFacingBtn1.setOnClickListener(this)
        mViewBinding.effectsBtn.setOnClickListener(this)
        mViewBinding.cameraTypeBtn.setOnClickListener(this)
        mViewBinding.settingsBtn.setOnClickListener(this)
        mViewBinding.voiceBtn.setOnClickListener(this)
        mViewBinding.resolutionBtn.setOnClickListener(this)
        mViewBinding.albumPreviewIv.setOnClickListener(this)
        mViewBinding.captureBtn.setOnViewClickListener(this)
        mViewBinding.albumPreviewIv.setTheme(PreviewImageView.Theme.DARK)
        mViewBinding.btnStartTextDetection.setOnClickListener {
            startTextDetection()
        }
        mViewBinding.btnStopTextDetection.setOnClickListener {
            stopAllDetections()
        }
        switchLayoutClick()
    }

    override fun initData() {
        super.initData()
        initObjectDetector()
        EventBus.with<Int>(BusKey.KEY_FRAME_RATE).observe(this) {
            mViewBinding.frameRateTv.text = "frame rate:  $it fps"
        }

        EventBus.with<Boolean>(BusKey.KEY_RENDER_READY).observe(this) { ready ->
            if (!ready) return@observe
            getDefaultEffect()?.apply {
                when (getClassifyId()) {
                    CameraEffect.CLASSIFY_ID_FILTER -> {
                        val filterId = -1
                        if (filterId != -99) {
                            removeRenderEffect(this)
                            mEffectDataList.find {
                                it.id == filterId
                            }?.also {
                                it.effect?.let { effect ->
                                    addRenderEffect(effect)
                                }
                            }
                        }
                    }
                    CameraEffect.CLASSIFY_ID_ANIMATION -> {
                        val animId = -1
                        if (animId != -99) {
                            removeRenderEffect(this)
                            mEffectDataList.find {
                                it.id == animId
                            }?.also {
                                it.effect?.let { effect ->
                                    addRenderEffect(effect)
                                }
                            }
                        }
                    }
                    else -> throw IllegalStateException("Unsupported classify")
                }
            }
        }
    }

    private fun startTextDetection() {
        if (!isTextRecognitionActive) {
            isTextRecognitionActive = true
            Toast.makeText(requireContext(), "Text Detection Started", Toast.LENGTH_SHORT).show()

            lifecycleScope.launch(Dispatchers.Default) {
                while (isTextRecognitionActive) {
                    captureFrameForTextDetection()
                    delay(DETECTION_INTERVAL_MS)
                }
            }
        }
    }

    private suspend fun captureFrameForTextDetection() {
        withContext(Dispatchers.Default) {
            val currentCamera = getCurrentCamera() as? CameraUVC ?: return@withContext
            currentCamera.captureImage(object : ICaptureCallBack {
                override fun onBegin() {}

                override fun onError(error: String?) {
                    ToastUtils.show("Error capturing frame: $error")
                }

                override fun onComplete(path: String?) {
                    path?.let {
                        val bitmap = BitmapFactory.decodeFile(it)
                        detectText(bitmap)

                        scheduleFileDeletion(it, 10_000L)
                    }
                }
            })
        }
    }

    private fun scheduleFileDeletion(filePath: String, delayMillis: Long) {
        Handler(Looper.getMainLooper()).postDelayed({
            val file = File(filePath)
            if (file.exists()) {

                val deleted = file.delete()
                if (deleted) {
                    removeImageFromGallery(filePath)
                    Log.d(TAG, "File deleted successfully: $filePath")
                } else {
                    Log.e(TAG, "Failed to delete file: $filePath")
                }
            }
        }, delayMillis)
    }

    private fun removeImageFromGallery(filePath: String) {

        val contentResolver = requireContext().contentResolver

        val uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val selection = "${MediaStore.Images.Media.DATA} = ?"
        val selectionArgs = arrayOf(filePath)

        val rowsDeleted = contentResolver.delete(uri, selection, selectionArgs)

        if (rowsDeleted > 0) {
            Log.d(TAG, "MediaStore updated successfully for file: $filePath")
        } else {
            Log.e(TAG, "Failed to update MediaStore for file: $filePath")
        }
    }

    private fun detectText(bitmap: Bitmap) {
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        textRecognizer.process(inputImage)
            .addOnSuccessListener { visionText ->
                val detectedText = visionText.text
                if (detectedText.isNotEmpty()) {
                    readDetectedText(detectedText)
                    updateOverlayViewForText(visionText.textBlocks, bitmap.height, bitmap.width)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Text Recognition Error: $e")
            }
    }

    private fun readDetectedText(text: String) {
        if (text.isEmpty() || !shouldSpeakNow()) {
            return
        }

        if (!textToSpeech.isSpeaking) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, text.hashCode().toString())
        } else {
            speechQueue.offer(text)
        }
    }

    private fun updateOverlayViewForText(textBlocks: List<com.google.mlkit.vision.text.Text.TextBlock>, imageHeight: Int, imageWidth: Int) {
        mViewBinding.overlay.setResults(
            emptyList(), imageHeight, imageWidth, textBlocks
        )
    }

    private fun stopAllDetections() {
        stopObjectDetection()
        stopTextDetection()
        Toast.makeText(requireContext(), "Detection Stopped", Toast.LENGTH_SHORT).show()
    }

    private fun stopTextDetection() {
        if (isTextRecognitionActive) {
            isTextRecognitionActive = false
            lifecycleScope.coroutineContext.cancelChildren()
            Toast.makeText(requireContext(), "Text Detection Stopped", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCameraState(
        self: MultiCameraClient.ICamera,
        code: ICameraStateCallBack.State,
        msg: String?
    ) {
        when (code) {
            ICameraStateCallBack.State.OPENED -> {
                handleCameraOpened()
            }
            ICameraStateCallBack.State.CLOSED -> handleCameraClosed()
            ICameraStateCallBack.State.ERROR -> handleCameraError(msg)
        }
    }

    private fun handleCameraError(msg: String?) {
        mViewBinding.uvcLogoIv.visibility = View.VISIBLE
        mViewBinding.frameRateTv.visibility = View.GONE
        ToastUtils.show("camera opened error: $msg")
    }

    private fun handleCameraClosed() {
        mViewBinding.uvcLogoIv.visibility = View.VISIBLE
        mViewBinding.frameRateTv.visibility = View.GONE
        ToastUtils.show("camera closed success")
        stopAllDetections()
    }

    private fun handleCameraOpened() {
        mViewBinding.uvcLogoIv.visibility = View.GONE
        mViewBinding.frameRateTv.visibility = View.VISIBLE
        mViewBinding.brightnessSb.max = (getCurrentCamera() as? CameraUVC)?.getBrightnessMax() ?: 100
        mViewBinding.brightnessSb.progress = (getCurrentCamera() as? CameraUVC)?.getBrightness() ?: 0
        Logger.i(TAG, "max = ${mViewBinding.brightnessSb.max}, progress = ${mViewBinding.brightnessSb.progress}")
        mViewBinding.brightnessSb.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                (getCurrentCamera() as? CameraUVC)?.setBrightness(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        ToastUtils.show("camera opened success")
    }

    private fun switchLayoutClick() {
        mViewBinding.takePictureModeTv.setOnClickListener {
            if (mCameraMode == CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC) {
                return@setOnClickListener
            }
            mCameraMode = CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC
            updateCameraModeSwitchUI()
        }
        mViewBinding.recordVideoModeTv.setOnClickListener {
            if (mCameraMode == CaptureMediaView.CaptureMode.MODE_CAPTURE_VIDEO) {
                return@setOnClickListener
            }
            mCameraMode = CaptureMediaView.CaptureMode.MODE_CAPTURE_VIDEO
            updateCameraModeSwitchUI()
        }
        mViewBinding.recordAudioModeTv.setOnClickListener {
            if (mCameraMode == CaptureMediaView.CaptureMode.MODE_CAPTURE_AUDIO) {
                return@setOnClickListener
            }
            mCameraMode = CaptureMediaView.CaptureMode.MODE_CAPTURE_AUDIO
            updateCameraModeSwitchUI()
        }
        updateCameraModeSwitchUI()
        showRecentMedia()
    }

    override fun getCameraView(): IAspectRatio {
        return AspectRatioTextureView(requireContext())
    }

    override fun getCameraViewContainer(): ViewGroup {
        return mViewBinding.cameraViewContainer
    }

    override fun getRootView(inflater: LayoutInflater, container: ViewGroup?): View {
        mViewBinding = FragmentUsbBinding.inflate(inflater, container, false)
        return mViewBinding.root
    }

    override fun getGravity(): Int = Gravity.CENTER

    override fun onViewClick(mode: CaptureMediaView.CaptureMode?) {
        if (!isCameraOpened()) {
            ToastUtils.show("camera not worked!")
            return
        }
        when (mode) {
            CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC -> captureImage()
            CaptureMediaView.CaptureMode.MODE_CAPTURE_AUDIO -> captureAudio()
            else -> captureVideo()
        }
    }

    private fun captureAudio() {
        if (isCapturingVideoOrAudio) {
            captureAudioStop()
            return
        }
        captureAudioStart(object : ICaptureCallBack {
            override fun onBegin() {
                isCapturingVideoOrAudio = true
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.DOING)
                mViewBinding.modeSwitchLayout.visibility = View.GONE
                mViewBinding.toolbarGroup.visibility = View.GONE
                mViewBinding.albumPreviewIv.visibility = View.GONE
                mViewBinding.lensFacingBtn1.visibility = View.GONE
                mViewBinding.recTimerLayout.visibility = View.VISIBLE
                startMediaTimer()
            }

            override fun onError(error: String?) {
                ToastUtils.show(error ?: "Unknown error")
                isCapturingVideoOrAudio = false
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.UNDO)
                stopMediaTimer()
            }

            override fun onComplete(path: String?) {
                isCapturingVideoOrAudio = false
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.UNDO)
                mViewBinding.modeSwitchLayout.visibility = View.VISIBLE
                mViewBinding.toolbarGroup.visibility = View.VISIBLE
                mViewBinding.albumPreviewIv.visibility = View.VISIBLE
                mViewBinding.lensFacingBtn1.visibility = View.VISIBLE
                mViewBinding.recTimerLayout.visibility = View.GONE
                stopMediaTimer()
                ToastUtils.show(path ?: "error")
            }
        })
    }

    private fun captureVideo() {
        if (isCapturingVideoOrAudio) {
            captureVideoStop()
            return
        }
        captureVideoStart(object : ICaptureCallBack {
            override fun onBegin() {
                isCapturingVideoOrAudio = true
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.DOING)
                mViewBinding.modeSwitchLayout.visibility = View.GONE
                mViewBinding.toolbarGroup.visibility = View.GONE
                mViewBinding.albumPreviewIv.visibility = View.GONE
                mViewBinding.lensFacingBtn1.visibility = View.GONE
                mViewBinding.recTimerLayout.visibility = View.VISIBLE
                startMediaTimer()
            }

            override fun onError(error: String?) {
                ToastUtils.show(error ?: "Unknown error")
                isCapturingVideoOrAudio = false
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.UNDO)
                stopMediaTimer()
            }

            override fun onComplete(path: String?) {
                ToastUtils.show(path ?: "")
                isCapturingVideoOrAudio = false
                mViewBinding.captureBtn.setCaptureVideoState(CaptureMediaView.CaptureVideoState.UNDO)
                mViewBinding.modeSwitchLayout.visibility = View.VISIBLE
                mViewBinding.toolbarGroup.visibility = View.VISIBLE
                mViewBinding.albumPreviewIv.visibility = View.VISIBLE
                mViewBinding.lensFacingBtn1.visibility = View.VISIBLE
                mViewBinding.recTimerLayout.visibility = View.GONE
                showRecentMedia(false)
                stopMediaTimer()
            }
        })
    }

    private fun captureImage() {
        captureImage(object : ICaptureCallBack {
            override fun onBegin() {
                mViewBinding.albumPreviewIv.showImageLoadProgress()
                mViewBinding.albumPreviewIv.setNewImageFlag(true)
            }

            override fun onError(error: String?) {
                ToastUtils.show(error ?: "Unknown error")
                mViewBinding.albumPreviewIv.cancelAnimation()
                mViewBinding.albumPreviewIv.setNewImageFlag(false)
            }

            override fun onComplete(path: String?) {
                showRecentMedia(true)
                mViewBinding.albumPreviewIv.setNewImageFlag(false)
            }
        })
    }

    override fun onDestroyView() {
        super.onDestroyView()
        mMoreMenu?.dismiss()
        tfLite.close()
        faceDetector.close()
        textRecognizer.close()
        textToSpeech.stop()
        textToSpeech.shutdown()
        stopAllDetections()
        stopLiveObjectDetection()
        inferenceDispatcher.close()
        speechRecognizer.destroy()
        previewDataCallback?.let {
            getCurrentCamera()?.removePreviewDataCallBack(it)
            previewDataCallback = null
        }
        mRecTimer?.cancel()
        mRecTimer = null
        lifecycleScope.coroutineContext.cancelChildren()
    }

    private fun stopObjectDetection() {
        if (isObjectDetectionActive) {
            isObjectDetectionActive = false
            previewDataCallback?.let {
                getCurrentCamera()?.removePreviewDataCallBack(it)
                previewDataCallback = null
            }
            Toast.makeText(requireContext(), "Object Detection Stopped", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onClick(v: View?) {
        if (!isCameraOpened()) {
            ToastUtils.show("camera not worked!")
            return
        }
        clickAnimation(v!!, object : AnimatorListenerAdapter() {
            override fun onAnimationEnd(animation: Animator) {
                when (v) {
                    mViewBinding.lensFacingBtn1 -> {
                        getCurrentCamera()?.let { strategy ->
                            if (strategy is CameraUVC) {
                                showUsbDevicesDialog(getDeviceList(), strategy.getUsbDevice())
                                return
                            }
                        }
                    }
                    mViewBinding.voiceBtn -> playMic()
                    mViewBinding.resolutionBtn -> showResolutionDialog()
                    mViewBinding.albumPreviewIv -> goToGallery()
                    else -> {

                    }
                }
            }
        })
    }

    private fun showUsbDevicesDialog(usbDeviceList: MutableList<UsbDevice>?, curDevice: UsbDevice?) {
        if (usbDeviceList.isNullOrEmpty()) {
            ToastUtils.show("Get usb device failed")
            return
        }
        val list = arrayListOf<String>()
        var selectedIndex: Int = -1
        for (index in usbDeviceList.indices) {
            val dev = usbDeviceList[index]
            val devName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && !dev.productName.isNullOrEmpty()) {
                "${dev.productName}(${curDevice?.deviceId})"
            } else {
                dev.deviceName
            }
            val curDevName = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && !curDevice?.productName.isNullOrEmpty()) {
                "${curDevice!!.productName}(${curDevice.deviceId})"
            } else {
                curDevice?.deviceName
            }
            if (devName == curDevName) {
                selectedIndex = index
            }
            list.add(devName)
        }
        MaterialDialog(requireContext()).show {
            listItemsSingleChoice(
                items = list,
                initialSelection = selectedIndex
            ) { _, index, _ ->
                if (selectedIndex == index) {
                    return@listItemsSingleChoice
                }
                switchCamera(usbDeviceList[index])
            }
        }
    }

    private fun showResolutionDialog() {
        mMoreMenu?.dismiss()
        getAllPreviewSizes().let { previewSizes ->
            if (previewSizes.isNullOrEmpty()) {
                ToastUtils.show("Get camera preview size failed")
                return
            }
            val list = arrayListOf<String>()
            var selectedIndex: Int = -1
            for (index in previewSizes.indices) {
                val w = previewSizes[index].width
                val h = previewSizes[index].height
                getCurrentPreviewSize()?.apply {
                    if (width == w && height == h) {
                        selectedIndex = index
                    }
                }
                list.add("$w x $h")
            }
            MaterialDialog(requireContext()).show {
                listItemsSingleChoice(
                    items = list,
                    initialSelection = selectedIndex
                ) { _, index, _ ->
                    if (selectedIndex == index) {
                        return@listItemsSingleChoice
                    }
                    updateResolution(previewSizes[index].width, previewSizes[index].height)
                }
            }
        }
    }

    private fun goToGallery() {
        try {
            Intent(Intent.ACTION_VIEW, MediaStore.Images.Media.EXTERNAL_CONTENT_URI).apply {
                startActivity(this)
            }
        } catch (e: Exception) {
            ToastUtils.show("open error: ${e.localizedMessage}")
        }
    }

    private fun playMic() {
        if (isPlayingMic) {
            stopPlayMic()
            return
        }
        startPlayMic(object : IPlayCallBack {
            override fun onBegin() {
                mViewBinding.voiceBtn.setImageResource(R.mipmap.camera_voice_on)
                isPlayingMic = true
            }

            override fun onError(error: String) {
                mViewBinding.voiceBtn.setImageResource(R.mipmap.camera_voice_off)
                isPlayingMic = false
            }

            override fun onComplete() {
                mViewBinding.voiceBtn.setImageResource(R.mipmap.camera_voice_off)
                isPlayingMic = false
            }
        })
    }

    private fun showRecentMedia(isImage: Boolean? = null) {

    }

    private fun updateCameraModeSwitchUI() {
        mViewBinding.modeSwitchLayout.children.forEach { it ->
            val tabTv = it as TextView
            val isSelected = tabTv.id == mCameraModeTabMap[mCameraMode]
            val typeface = if (isSelected) Typeface.BOLD else Typeface.NORMAL
            tabTv.typeface = Typeface.defaultFromStyle(typeface)
            if (isSelected) {
                0xFFFFFFFF
            } else {
                0xFFD7DAE1
            }.also {
                tabTv.setTextColor(it.toInt())
            }
            tabTv.setShadowLayer(
                Utils.dp2px(requireContext(), 1F).toFloat(),
                0F,
                0F,
                0xBF000000.toInt()
            )

            if (isSelected) {
                R.mipmap.camera_preview_dot_blue
            } else {
                R.drawable.camera_bottom_dot_transparent
            }.also {
                TextViewCompat.setCompoundDrawablesRelativeWithIntrinsicBounds(tabTv, 0, 0, 0, it)
            }
            tabTv.compoundDrawablePadding = 1
        }
        mViewBinding.captureBtn.setCaptureViewTheme(CaptureMediaView.CaptureViewTheme.THEME_WHITE)
        val height = mViewBinding.controlPanelLayout.height
        mViewBinding.captureBtn.setCaptureMode(mCameraMode)
        if (mCameraMode == CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC) {
            val translationX = ObjectAnimator.ofFloat(
                mViewBinding.controlPanelLayout,
                "translationY",
                height.toFloat(),
                0.0f
            )
            translationX.duration = 600
            translationX.addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationStart(animation: Animator) {
                    super.onAnimationStart(animation)
                    mViewBinding.controlPanelLayout.visibility = View.VISIBLE
                }
            })
            translationX.start()
        } else {
            val translationX = ObjectAnimator.ofFloat(
                mViewBinding.controlPanelLayout,
                "translationY",
                0.0f,
                height.toFloat()
            )
            translationX.duration = 600
            translationX.addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    super.onAnimationEnd(animation)
                    mViewBinding.controlPanelLayout.visibility = View.INVISIBLE
                }
            })
            translationX.start()
        }
    }

    private fun clickAnimation(v: View, listener: Animator.AnimatorListener) {
        val scaleXAnim: ObjectAnimator = ObjectAnimator.ofFloat(v, "scaleX", 1.0f, 0.4f, 1.0f)
        val scaleYAnim: ObjectAnimator = ObjectAnimator.ofFloat(v, "scaleY", 1.0f, 0.4f, 1.0f)
        val alphaAnim: ObjectAnimator = ObjectAnimator.ofFloat(v, "alpha", 1.0f, 0.4f, 1.0f)
        val animatorSet = AnimatorSet()
        animatorSet.duration = 150
        animatorSet.addListener(listener)
        animatorSet.playTogether(scaleXAnim, scaleYAnim, alphaAnim)
        animatorSet.start()
    }

    private fun startMediaTimer() {
        mRecTimer?.cancel()
        mRecTimer = Timer().apply {
            schedule(object : TimerTask() {
                override fun run() {
                    mRecSeconds++
                    if (mRecSeconds >= 60) {
                        mRecSeconds = 0
                        mRecMinute++
                    }
                    if (mRecMinute >= 60) {
                        mRecMinute = 0
                        mRecHours++
                        if (mRecHours >= 24) {
                            mRecHours = 0
                            mRecMinute = 0
                            mRecSeconds = 0
                        }
                    }
                    mMainHandler.sendEmptyMessage(WHAT_START_TIMER)
                }
            }, 1000, 1000)
        }
    }

    override fun onDestroy() {
        textToSpeech.stop()
        textToSpeech.shutdown()
        textRecognizer.close()
        super.onDestroy()
    }

    private fun stopMediaTimer() {
        mRecTimer?.cancel()
        mRecTimer = null
        mRecHours = 0
        mRecMinute = 0
        mRecSeconds = 0
        mMainHandler.sendEmptyMessage(WHAT_STOP_TIMER)
    }

    private fun calculateTime(seconds: Int, minute: Int, hour: Int? = null): String {
        return buildString {
            if (hour != null) {
                append(if (hour < 10) "0$hour" else hour.toString())
                append(":")
            }
            append(if (minute < 10) "0$minute" else minute.toString())
            append(":")
            append(if (seconds < 10) "0$seconds" else seconds.toString())
        }
    }

    companion object {
        private const val TAG = "UsbFragment"
        private const val WHAT_START_TIMER = 0x00
        private const val WHAT_STOP_TIMER = 0x01
        private const val DETECTION_INTERVAL_MS = 2000L
    }
}