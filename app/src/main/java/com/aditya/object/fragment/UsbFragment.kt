package com.aditya.`object`.fragment

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.YuvImage
import android.hardware.usb.UsbDevice
import android.os.*
import android.provider.MediaStore
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupWindow
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.view.children
import androidx.core.widget.TextViewCompat
import androidx.lifecycle.lifecycleScope
import com.aditya.`object`.ObjectDetectorHelper
import com.aditya.`object`.R
import com.aditya.`object`.databinding.FragmentUsbBinding
import com.afollestad.materialdialogs.MaterialDialog
import com.afollestad.materialdialogs.list.listItemsSingleChoice
import com.google.mlkit.vision.common.InputImage
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
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.ByteArrayOutputStream
import java.io.File
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
    private val inferenceDispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val speechQueue: Queue<String> = LinkedList()
    private var lastSpokenTime: Long = 0
    private val MIN_SPEECH_INTERVAL_MS = 2000L
    private val SPEECH_RECOGNITION_DELAY = 1000L

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private var isListening: Boolean = false


    private var previousDetections: MutableSet<String> = mutableSetOf()


    private val mCameraModeTabMap = mapOf(
        CaptureMediaView.CaptureMode.MODE_CAPTURE_PIC to R.id.takePictureModeTv,
        CaptureMediaView.CaptureMode.MODE_CAPTURE_VIDEO to R.id.recordVideoModeTv,
        CaptureMediaView.CaptureMode.MODE_CAPTURE_AUDIO to R.id.recordAudioModeTv
    )


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
            }
            else -> {
                Log.d(TAG, "Unknown command: $command")
            }
        }
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

        initObjectDetector()

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

            getCurrentCamera()?.addPreviewDataCallBack(object : IPreviewDataCallBack {
                override fun onPreviewData(
                    data: ByteArray?,
                    width: Int,
                    height: Int,
                    format: IPreviewDataCallBack.DataFormat
                ) {
                    Log.d(TAG, "Preview Data Received: Width = $width, Height = $height, Format = $format")
                    processFrame(data, width, height, format)
                }
            })
        }
    }


    private fun processFrame(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {
        data ?: return

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
                                updateOverlayView(detections, imageHeight, imageWidth)
                                handleDetections(detections)
                            } else {
                                Log.d(TAG, "No detections found.")
                            }
                        }
                    }
                }
            )
        }
    }

    private fun handleDetections(detections: List<Detection>) {
        val currentDetections = mutableSetOf<String>()
        for (detection in detections) {
            detection.categories.firstOrNull()?.label?.let { currentDetections.add(it) }
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

    private fun updateOverlayView(detections: List<Detection>, imageHeight: Int, imageWidth: Int) {
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
        if (!isTextRecognitionActive) {  // Check if already active
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
            lifecycleScope.coroutineContext.cancelChildren()  // Cancels all child coroutines
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
        stopAllDetections()
        stopLiveObjectDetection()
        inferenceDispatcher.close()
        speechRecognizer.destroy()

        lifecycleScope.coroutineContext.cancelChildren()
    }

    private fun stopObjectDetection() {
        if (isObjectDetectionActive) {
            isObjectDetectionActive = false
            getCurrentCamera()?.removePreviewDataCallBack(object : IPreviewDataCallBack {
                override fun onPreviewData(data: ByteArray?, width: Int, height: Int, format: IPreviewDataCallBack.DataFormat) {}
            })
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
                        // No need to trigger detection manually
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
        // Implement logic to show recent media if needed
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