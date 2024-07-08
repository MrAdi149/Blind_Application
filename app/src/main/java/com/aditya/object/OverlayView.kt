package com.aditya.`object`

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.aditya.`object`.R
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection
import com.google.mlkit.vision.text.Text

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList<Detection>()
    private var textBlocks: List<Text.TextBlock>? = null
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
//    private var textPaint = Paint()

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textBlocks = LinkedList()
        results = LinkedList()
        invalidate()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    private val paint: Paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val textPaint: Paint = Paint().apply {
        color = Color.WHITE
        textSize = 48f
        style = Paint.Style.FILL
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw detected objects
        for (result in results) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText = "${result.categories[0].label} ${String.format("%.2f", result.categories[0].score)}"

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }

        // Draw detected text blocks
        textBlocks?.forEach { textBlock ->
            textBlock.boundingBox?.let { rect ->
                val scaledRect = scaleRect(rect, scaleFactor)
                canvas.drawRect(scaledRect, paint)

                // Draw each line of text within the text block
                textBlock.lines.forEach { line ->
                    val lineRect = scaleRect(line.boundingBox!!, scaleFactor)
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
        detectionResults: List<Detection>,
        imageHeight: Int,
        imageWidth: Int,
        textBlocks: List<Text.TextBlock>? = null
    ) {
        results = detectionResults
        this.textBlocks = textBlocks ?: LinkedList()

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
