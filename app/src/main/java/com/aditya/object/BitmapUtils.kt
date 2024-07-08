package com.aditya.`object`

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object BitmapUtils {

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv_420_888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun yuv_420_888ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4
        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        var rowStride = image.planes[0].rowStride
        assert(image.planes[0].pixelStride == 1)

        var pos = 0

        if (rowStride == width) { // likely
            yBuffer[nv21, 0, ySize]
            pos += ySize
        } else {
            var yPos = 0
            for (i in 0 until height) {
                yBuffer.position(yPos)
                yBuffer[nv21, pos, width]
                pos += width
                yPos += rowStride
            }
        }

        rowStride = image.planes[2].rowStride
        val pixelStride = image.planes[2].pixelStride
        assert(rowStride == image.planes[1].rowStride)
        assert(pixelStride == image.planes[1].pixelStride)

        if (pixelStride == 2 && rowStride == width && uvSize + ySize == nv21.size) {
            vBuffer[nv21, ySize, width * height / 2]
        } else {
            for (row in 0 until height / 2) {
                var uvPos = row * rowStride
                for (col in 0 until width / 2) {
                    nv21[pos++] = vBuffer.get(uvPos)
                    nv21[pos++] = uBuffer.get(uvPos)
                    uvPos += pixelStride
                }
            }
        }
        return nv21
    }
}
