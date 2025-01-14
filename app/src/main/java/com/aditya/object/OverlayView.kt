package com.aditya.`object`

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.aditya.`object`.R
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection
import com.google.mlkit.vision.text.Text

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<DetectionResult> = LinkedList()
    private var textBlocks: List<Text.TextBlock>? = null
    private var objectBoxPaint = Paint()
    private var objectTextBackgroundPaint = Paint()
    private var objectTextPaint = Paint()
    private var textBlockBoxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var descriptionBoxPaint = Paint()
    private var descriptionTextPaint = Paint()
    private var description: String? = null
    private var describedObject: Detection? = null

    private var scaleFactor: Float = 1f
    private var objectCountPaint = Paint()
    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textBlocks = LinkedList()
        results = LinkedList()
        description = null
        describedObject = null
        invalidate()
    }

    private fun initPaints() {
        // Paint for object bounding box
        objectBoxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        objectBoxPaint.strokeWidth = 10F
        objectBoxPaint.style = Paint.Style.STROKE
        objectBoxPaint.setShadowLayer(10f, 0f, 0f, Color.BLACK)

        // Paint for object text background
        objectTextBackgroundPaint.color = Color.argb(150, 0, 0, 0) // Semi-transparent black
        objectTextBackgroundPaint.style = Paint.Style.FILL

        // Paint for object text
        objectTextPaint.color = Color.WHITE
        objectTextPaint.textSize = 60f
        objectTextPaint.style = Paint.Style.FILL_AND_STROKE
        objectTextPaint.strokeWidth = 2f
        objectTextPaint.isAntiAlias = true
        objectTextPaint.setShadowLayer(5.0f, 0.0f, 0.0f, Color.BLACK)

        // Paint for object count display
        objectCountPaint.color = Color.WHITE
        objectCountPaint.textSize = 80f
        objectCountPaint.style = Paint.Style.FILL_AND_STROKE
        objectCountPaint.strokeWidth = 2f
        objectCountPaint.isAntiAlias = true
        objectCountPaint.setShadowLayer(5.0f, 0.0f, 0.0f, Color.BLACK)

        // Paint for text block bounding box
        textBlockBoxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        textBlockBoxPaint.strokeWidth = 5F
        textBlockBoxPaint.style = Paint.Style.STROKE

        // Paint for text background
        textBackgroundPaint.color = Color.argb(150, 0, 0, 0) // Semi-transparent black
        textBackgroundPaint.style = Paint.Style.FILL

        // Paint for text
        textPaint.color = Color.WHITE
        textPaint.textSize = 48f
        textPaint.style = Paint.Style.FILL
        textPaint.isAntiAlias = true
        textPaint.setShadowLayer(2.0f, 0.0f, 0.0f, Color.BLACK)

        // Paint for description box
        descriptionBoxPaint.color = Color.argb(200, 0, 0, 0) // More opaque black
        descriptionBoxPaint.style = Paint.Style.FILL

        // Paint for description text
        descriptionTextPaint.color = Color.YELLOW
        descriptionTextPaint.textSize = 60f
        descriptionTextPaint.style = Paint.Style.FILL_AND_STROKE
        descriptionTextPaint.strokeWidth = 2f
        descriptionTextPaint.isAntiAlias = true
        descriptionTextPaint.setShadowLayer(5.0f, 0.0f, 0.0f, Color.BLACK)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        Log.d("OverlayView", "onDraw called")

        // Draw the object count on the left side of the screen
        val objectCountText = "Objects: ${results.size}"
        canvas.drawText(objectCountText, 20f, 100f, objectCountPaint)

        // Group objects by their labels and count them
        val groupedObjects = results.groupBy { it.detection.categories.firstOrNull()?.label ?: "Unknown" }
        var yOffset = 200f // Start drawing the counts below the total object count

        groupedObjects.forEach { (label, objects) ->
            val objectCountByLabel = "$label: ${objects.size}"
            canvas.drawText(objectCountByLabel, 20f, yOffset, objectCountPaint)
            yOffset += 100f // Move down for the next label
        }

        // Draw detected objects
        for (resultWithDistance in results) {
            val detection = resultWithDistance.detection
            val distance = resultWithDistance.distance
            val boundingBox = detection.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRoundRect(drawableRect, 32f, 32f, objectBoxPaint)

            val drawableText = "${detection.categories[0].label} (${String.format("%.2f", distance)}m)"

            objectTextBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRoundRect(
                RectF(left, top, left + textWidth + BOUNDING_RECT_TEXT_PADDING * 2, top + textHeight + BOUNDING_RECT_TEXT_PADDING * 2),
                16f,
                16f,
                objectTextBackgroundPaint
            )

            canvas.drawText(drawableText, left + BOUNDING_RECT_TEXT_PADDING, top + textHeight + BOUNDING_RECT_TEXT_PADDING, objectTextPaint)
        }

        // Draw detected text blocks
        textBlocks?.forEach { textBlock ->
            textBlock.boundingBox?.let { rect ->
                val scaledRect = scaleRect(rect, scaleFactor)
                Log.d("OverlayView", "Drawing text block bounding box at: ${scaledRect.left}, ${scaledRect.top}, ${scaledRect.right}, ${scaledRect.bottom}")
                canvas.drawRoundRect(RectF(scaledRect), 16f, 16f, textBlockBoxPaint)

                // Draw each line of text within the text block
                textBlock.lines.forEach { line ->
                    val lineRect = scaleRect(line.boundingBox!!, scaleFactor)
                    canvas.drawRoundRect(RectF(lineRect), 16f, 16f, textBackgroundPaint)
                    canvas.drawText(line.text, lineRect.left.toFloat(), lineRect.bottom.toFloat(), textPaint)
                }
            }
        }
    }

    private fun scaleRect(rect: Rect, scaleFactor: Float): Rect {
        return Rect(
            (rect.left * scaleFactor).toInt(),
            (rect.top * scaleFactor).toInt(),
            (rect.right * scaleFactor).toInt(),
            (rect.bottom * scaleFactor).toInt()
        )
    }

    fun setResults(
        detectionResults: List<DetectionResult>,
        imageHeight: Int,
        imageWidth: Int,
        textBlocks: List<Text.TextBlock>? = null
    ) {
        results = detectionResults
        this.textBlocks = textBlocks ?: LinkedList()

        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
        invalidate()
    }

    fun describeObject(detection: Detection, description: String) {
        describedObject = detection
        this.description = description
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}