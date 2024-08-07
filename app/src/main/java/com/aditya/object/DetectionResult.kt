package com.aditya.`object`

import org.tensorflow.lite.task.vision.detector.Detection

data class DetectionResult(
    val detection: Detection,
    val distance: Float
)

