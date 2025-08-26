package com.google.mediapipe.examples.facedetection

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult
import kotlinx.coroutines.*
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: FaceDetectorResult? = null
    private var boxPaint = Paint()
    private var eyePaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var fullBitmap: Bitmap? = null
    private var originalImageHeight: Int = 0
    private var originalImageWidth: Int = 0
    private var bounds = Rect()
    private var uniformScaleFactor = 1f
    private var xOffset = 0f
    private var yOffset = 0f

    private var cachedFaceBitmaps = HashMap<FaceRect, Bitmap>()
    private var cachedEyeRects = HashMap<FaceRect, List<RectF>>()
    private var rotationDegrees = 0f
    private lateinit var eyeCascade: CascadeClassifier
    
    // Contrast detection and heatmap variables
    private var previousFrame: Mat? = null
    private var heatmapData = HashMap<FaceRect, FloatArray>()
    private var heatmapAge = HashMap<FaceRect, Long>()
    private var lastFaceRegions = mutableListOf<RectF>()
    private val heatmapDecayTime = 5000L // 5 seconds
    private val maxHeatmapValue = 100f
    private var dynamicContrastThreshold = 30.0

    private val backgroundExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    private val ioScope = CoroutineScope(Dispatchers.IO)
    private var processingJob: Job? = null
    private var contrastJob: Job? = null
    private val lock = java.util.concurrent.locks.ReentrantLock()

    init {
        initPaints()
        System.loadLibrary("opencv_java4")
        initializeEyeCascade()
    }

    private fun initializeEyeCascade() {
        try {
            val rawResource = context?.resources?.openRawResource(R.raw.haarcascade_eye)
            val cascadeDir = context?.getDir("cascade", Context.MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, "haarcascade_eye.xml")
            val outputStream = FileOutputStream(cascadeFile)

            rawResource?.let {
                val buffer = ByteArray(4096)
                var bytesRead = it.read(buffer)
                while (bytesRead != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    bytesRead = it.read(buffer)
                }
                it.close()
            }
            outputStream.close()

            eyeCascade = CascadeClassifier(cascadeFile.absolutePath)
            cascadeFile.delete()
            cascadeDir?.delete()

            if (eyeCascade.empty()) {
                Log.e("OverlayView", "Failed to load cascade classifier")
                eyeCascade = CascadeClassifier()
            }
        } catch (e: Exception) {
            Log.e("OverlayView", "Error loading cascade classifier: ${e.message}")
            eyeCascade = CascadeClassifier()
        }
    }

    private data class FaceRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
        override fun toString(): String {
            return "{$left,$top,$right,$bottom}"
        }
    }

    fun clear() {
        lock.lock()
        try {
            results = null
            cachedFaceBitmaps.clear()
            cachedEyeRects.clear()
            heatmapData.clear()
            heatmapAge.clear()
            lastFaceRegions.clear()
            previousFrame?.release()
            previousFrame = null
            textPaint.reset()
            textBackgroundPaint.reset()
            boxPaint.reset()
            eyePaint.reset()
        } finally {
            lock.unlock()
        }
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.mp_primary)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        eyePaint.color = Color.GREEN
        eyePaint.strokeWidth = 4F
        eyePaint.style = Paint.Style.STROKE
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        recalculateScaleAndOffsets()
    }

    private fun recalculateScaleAndOffsets() {
        if (originalImageWidth == 0 || originalImageHeight == 0) return

        val availableWidth = width.toFloat()
        val availableHeight = height.toFloat()

        val scaleFactorX = availableWidth / originalImageWidth
        val scaleFactorY = availableHeight / originalImageHeight
        uniformScaleFactor = maxOf(scaleFactorX, scaleFactorY)

        val scaledImageWidth = originalImageWidth * uniformScaleFactor
        val scaledImageHeight = originalImageHeight * uniformScaleFactor
        xOffset = (availableWidth - scaledImageWidth) / 2
        yOffset = (availableHeight - scaledImageHeight) / 2
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        lock.lock()
        try {
            // Decay heatmap data
            decayHeatmap()
            
            results?.let {
                for (detection in it.detections()) {
                    val boundingBox = detection.boundingBox()

                    val scaledLeft = (boundingBox.left * uniformScaleFactor) + xOffset
                    val scaledTop = (boundingBox.top * uniformScaleFactor) + yOffset
                    val scaledRight = (boundingBox.right * uniformScaleFactor) + xOffset
                    val scaledBottom = (boundingBox.bottom * uniformScaleFactor) + yOffset

                    val drawableRect = RectF(scaledLeft, scaledTop, scaledRight, scaledBottom)

                    val rectKey = FaceRect(
                        scaledLeft.roundToInt(),
                        scaledTop.roundToInt(),
                        scaledRight.roundToInt(),
                        scaledBottom.roundToInt()
                    )
                    
                    // Draw heatmap first (behind everything else)
                    heatmapData[rectKey]?.let { heatmap ->
                        drawHeatmap(canvas, rectKey, heatmap)
                    }
                    
                    val cachedBitmap = cachedFaceBitmaps[rectKey]
                    cachedBitmap?.let {
                        canvas.drawBitmap(it, scaledLeft, scaledTop, null)
                    }

                    // Draw the detection box
                    canvas.drawRect(drawableRect, boxPaint)

                    // Draw detected eyes
                    cachedEyeRects[rectKey]?.forEach { eyeRect ->
                        canvas.drawRect(eyeRect, eyePaint)
                    }

                    // Draw detection label
                    val drawableText = detection.categories()[0].categoryName() + " " +
                            String.format("%.2f", detection.categories()[0].score())

                    textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                    val textWidth = bounds.width().toFloat()
                    val textHeight = bounds.height().toFloat()

                    canvas.drawRect(
                        scaledLeft,
                        scaledTop,
                        scaledLeft + textWidth,
                        scaledTop + textHeight,
                        textBackgroundPaint
                    )

                    canvas.drawText(drawableText, scaledLeft, scaledTop + textHeight, textPaint)
                }
            }
        } finally {
            lock.unlock()
        }
    }

    private fun applyTransformations(bitmap: Bitmap, drawableRect: RectF): Bitmap {
        val bitmapMatrix = Matrix()
        if (rotationDegrees != 0f) {
            bitmapMatrix.postRotate(rotationDegrees)
        }

        val scaleX = drawableRect.width() / bitmap.width
        val scaleY = drawableRect.height() / bitmap.height
        bitmapMatrix.postScale(scaleX, scaleY)
        
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, bitmapMatrix, true)
    }

    private fun detectEyes(grayBitmap: Bitmap): List<RectF> {
        val eyeRects = mutableListOf<RectF>()
        
        val grayMat = Mat()
        Utils.bitmapToMat(grayBitmap, grayMat)
        
        val eyes = MatOfRect()
        eyeCascade.detectMultiScale(
            grayMat,
            eyes,
            1.1,
            2,
            0,
            Size(30.0, 30.0),
            Size(grayBitmap.width.toDouble(), grayBitmap.height.toDouble())
        )

        val eyeArray = eyes.toArray()
        for (eye in eyeArray) {
            eyeRects.add(RectF(
                eye.x.toFloat(),
                eye.y.toFloat(),
                (eye.x + eye.width).toFloat(),
                (eye.y + eye.height).toFloat()
            ))
        }

        grayMat.release()
        eyes.release()

        return eyeRects
    }

    private fun toGrayscale(faceBitmap: Bitmap): Bitmap {
        val bmpMonochrome = Bitmap.createBitmap(
            faceBitmap.width,
            faceBitmap.height,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bmpMonochrome)
        val ma = ColorMatrix()
        ma.setSaturation(0f)
        val paint = Paint()
        paint.colorFilter = ColorMatrixColorFilter(ma)
        canvas.drawBitmap(faceBitmap, 0f, 0f, paint)
        return bmpMonochrome
    }

    private fun getRotationDegrees(): Float {
        val rotation = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            context?.display?.rotation ?: 0
        } else {
            (context?.getSystemService(Context.WINDOW_SERVICE) as? android.view.WindowManager)?.defaultDisplay?.rotation ?: 0
        }
        return when (rotation) {
            android.view.Surface.ROTATION_90 -> 90f
            android.view.Surface.ROTATION_180 -> 180f
            android.view.Surface.ROTATION_270 -> 270f
            else -> 0f
        }
    }

    fun setResults(
        detectionResults: FaceDetectorResult,
        imageHeight: Int,
        imageWidth: Int,
        bitmap: Bitmap
    ) {
        processingJob?.cancel()

        lock.lock()
        try {
            cachedFaceBitmaps.clear()
            cachedEyeRects.clear()
            results = detectionResults
            fullBitmap = bitmap
            originalImageHeight = imageHeight
            originalImageWidth = imageWidth
            rotationDegrees = getRotationDegrees()
            
            // Store face regions for contrast detection
            lastFaceRegions.clear()
            for (detection in detectionResults.detections()) {
                val boundingBox = detection.boundingBox()
                lastFaceRegions.add(RectF(
                    boundingBox.left,
                    boundingBox.top,
                    boundingBox.right,
                    boundingBox.bottom
                ))
            }
        } finally {
            lock.unlock()
        }

        recalculateScaleAndOffsets()

        processingJob = ioScope.launch {
            processFacesInBackground()
            withContext(Dispatchers.Main) {
                invalidate()
            }
        }
    }

    private suspend fun processFacesInBackground() {
        lock.lock()
        try {
            val currentResults = results
            val currentBitmap = fullBitmap
            if (currentResults == null || currentBitmap == null) return
            val newCache = HashMap<FaceRect, Bitmap>()
            val newEyeRects = HashMap<FaceRect, List<RectF>>()

            for (detection in currentResults.detections()) {
                val boundingBox = detection.boundingBox()

                val scaledLeft = (boundingBox.left * uniformScaleFactor) + xOffset
                val scaledTop = (boundingBox.top * uniformScaleFactor) + yOffset
                val scaledRight = (boundingBox.right * uniformScaleFactor) + xOffset
                val scaledBottom = (boundingBox.bottom * uniformScaleFactor) + yOffset
                val drawableRect = RectF(scaledLeft, scaledTop, scaledRight, scaledBottom)

                val rectKey = FaceRect(
                    scaledLeft.roundToInt(),
                    scaledTop.roundToInt(),
                    scaledRight.roundToInt(),
                    scaledBottom.roundToInt()
                )

                var cachedBitmap = cachedFaceBitmaps[rectKey]

                if (cachedBitmap == null) {
                    currentBitmap.let { bitmap ->
                        val xInt = boundingBox.left.roundToInt().coerceAtLeast(0)
                        val yInt = boundingBox.top.roundToInt().coerceAtLeast(0)
                        val widthInt = (boundingBox.right - boundingBox.left).roundToInt()
                            .coerceAtMost(bitmap.width - xInt)
                        val heightInt = (boundingBox.bottom - boundingBox.top).roundToInt()
                            .coerceAtMost(bitmap.height - yInt)

                        if (widthInt > 0 && heightInt > 0 && xInt < bitmap.width && yInt < bitmap.height) {
                            try {
                                val faceBitmap = Bitmap.createBitmap(
                                    bitmap,
                                    xInt,
                                    yInt,
                                    widthInt,
                                    heightInt
                                )
                                val grayFaceBitmap = toGrayscale(faceBitmap)

                                grayFaceBitmap?.let { grayBitmap ->
                                    val scaledBitmap = applyTransformations(grayBitmap, drawableRect)
                                    newCache[rectKey] = scaledBitmap
                                    
                                    // Detect eyes in the grayscale face bitmap
                                    val eyeRects = detectEyes(grayBitmap)
                                    newEyeRects[rectKey] = eyeRects.map { eyeRect ->
                                        RectF(
                                            scaledLeft + (eyeRect.left * uniformScaleFactor),
                                            scaledTop + (eyeRect.top * uniformScaleFactor),
                                            scaledLeft + (eyeRect.right * uniformScaleFactor),
                                            scaledTop + (eyeRect.bottom * uniformScaleFactor)
                                        )
                                    }
                                    
                                    grayBitmap.recycle()
                                }
                                faceBitmap.recycle()
                            } catch (e: IllegalArgumentException) {
                                Log.e("OverlayView", "Bitmap creation failed: ${e.message}")
                            }
                        }
                    }
                } else {
                    newCache[rectKey] = cachedBitmap
                    cachedEyeRects[rectKey]?.let { eyeRects ->
                        newEyeRects[rectKey] = eyeRects
                    }
                }
            }

            cachedFaceBitmaps = newCache
            cachedEyeRects = newEyeRects
        } finally {
            lock.unlock()
        }
    }
    
    // Process contrast detection for current frame
    fun processContrastDetection(bitmap: Bitmap) {
        contrastJob?.cancel()
        contrastJob = ioScope.launch {
            performContrastDetection(bitmap)
            withContext(Dispatchers.Main) {
                invalidate()
            }
        }
    }
    
    private suspend fun performContrastDetection(currentBitmap: Bitmap) {
        lock.lock()
        try {
            if (lastFaceRegions.isEmpty()) return
            
            val currentMat = Mat()
            Utils.bitmapToMat(currentBitmap, currentMat)
            
            val grayCurrentMat = Mat()
            Imgproc.cvtColor(currentMat, grayCurrentMat, Imgproc.COLOR_RGB2GRAY)
            
            if (previousFrame != null) {
                for (faceRegion in lastFaceRegions) {
                    val faceRect = Rect(
                        max(0, faceRegion.left.toInt()),
                        max(0, faceRegion.top.toInt()),
                        min(currentBitmap.width, faceRegion.right.toInt()),
                        min(currentBitmap.height, faceRegion.bottom.toInt())
                    )
                    
                    if (faceRect.width > 0 && faceRect.height > 0) {
                        val faceKey = FaceRect(faceRect.x, faceRect.y, faceRect.x + faceRect.width, faceRect.y + faceRect.height)
                        
                        // Extract face region from current and previous frames
                        val currentFace = Mat(grayCurrentMat, faceRect)
                        val previousFace = Mat(previousFrame!!, faceRect)
                        
                        // Calculate frame difference
                        val diff = Mat()
                        Core.absdiff(currentFace, previousFace, diff)
                        
                        // Apply dynamic threshold
                        val threshold = Mat()
                        val adaptiveThreshold = calculateDynamicThreshold(diff)
                        Imgproc.threshold(diff, threshold, adaptiveThreshold, 255.0, Imgproc.THRESH_BINARY)
                        
                        // Find contours
                        val contours = mutableListOf<MatOfPoint>()
                        val hierarchy = Mat()
                        Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
                        
                        // Update heatmap data
                        updateHeatmapData(faceKey, contours, faceRect)
                        
                        // Clean up
                        currentFace.release()
                        previousFace.release()
                        diff.release()
                        threshold.release()
                        hierarchy.release()
                        contours.forEach { it.release() }
                    }
                }
            }
            
            // Store current frame as previous for next iteration
            previousFrame?.release()
            previousFrame = grayCurrentMat.clone()
            
            currentMat.release()
            grayCurrentMat.release()
            
        } finally {
            lock.unlock()
        }
    }
    
    private fun calculateDynamicThreshold(diff: Mat): Double {
        val mean = Core.mean(diff)
        val meanValue = mean.`val`[0]
        
        // Adjust threshold based on overall image activity
        dynamicContrastThreshold = when {
            meanValue < 10 -> 15.0  // Low activity - lower threshold
            meanValue < 30 -> 25.0  // Medium activity - medium threshold
            else -> 40.0            // High activity - higher threshold
        }
        
        return dynamicContrastThreshold
    }
    
    private fun updateHeatmapData(faceKey: FaceRect, contours: List<MatOfPoint>, faceRect: Rect) {
        val currentTime = System.currentTimeMillis()
        val width = faceRect.width
        val height = faceRect.height
        
        // Initialize heatmap array if it doesn't exist
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(width * height) { 0f }
            heatmapData[faceKey] = heatmap
        }
        
        // Add heat for each contour
        for (contour in contours) {
            val points = contour.toArray()
            for (point in points) {
                val x = point.x.toInt()
                val y = point.y.toInt()
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    val index = y * width + x
                    if (index < heatmap.size) {
                        heatmap[index] = min(maxHeatmapValue, heatmap[index] + 5f)
                    }
                }
            }
        }
        
        heatmapAge[faceKey] = currentTime
    }
    
    private fun decayHeatmap() {
        val currentTime = System.currentTimeMillis()
        val iterator = heatmapData.entries.iterator()
        
        while (iterator.hasNext()) {
            val entry = iterator.next()
            val age = heatmapAge[entry.key] ?: currentTime
            
            if (currentTime - age > heatmapDecayTime) {
                iterator.remove()
                heatmapAge.remove(entry.key)
            } else {
                // Decay existing heat values
                val decayRate = 0.98f
                for (i in entry.value.indices) {
                    entry.value[i] *= decayRate
                }
            }
        }
    }
    
    private fun drawHeatmap(canvas: Canvas, faceKey: FaceRect, heatmap: FloatArray) {
        val width = faceKey.right - faceKey.left
        val height = faceKey.bottom - faceKey.top
        
        if (width <= 0 || height <= 0) return
        
        val paint = Paint()
        paint.style = Paint.Style.FILL
        
        for (y in 0 until height step 4) { // Sample every 4 pixels for performance
            for (x in 0 until width step 4) {
                val index = y * width + x
                if (index < heatmap.size) {
                    val intensity = heatmap[index] / maxHeatmapValue
                    if (intensity > 0.1f) { // Only draw if there's significant heat
                        val color = getHeatmapColor(intensity)
                        paint.color = color
                        
                        val screenX = faceKey.left + x
                        val screenY = faceKey.top + y
                        canvas.drawRect(
                            (screenX * uniformScaleFactor) + xOffset,
                            (screenY * uniformScaleFactor) + yOffset,
                            ((screenX + 4) * uniformScaleFactor) + xOffset,
                            ((screenY + 4) * uniformScaleFactor) + yOffset,
                            paint
                        )
                    }
                }
            }
        }
    }
    
    private fun getHeatmapColor(intensity: Float): Int {
        // Create a color gradient from blue (cold) to red (hot) with transparency
        val alpha = (intensity * 120).toInt().coerceIn(0, 120) // Semi-transparent
        
        return when {
            intensity < 0.3f -> {
                val blue = (255 * (intensity / 0.3f)).toInt()
                Color.argb(alpha, 0, 0, blue)
            }
            intensity < 0.6f -> {
                val green = (255 * ((intensity - 0.3f) / 0.3f)).toInt()
                Color.argb(alpha, 0, green, 255)
            }
            intensity < 0.8f -> {
                val red = (255 * ((intensity - 0.6f) / 0.2f)).toInt()
                val green = (255 * (1f - ((intensity - 0.6f) / 0.2f))).toInt()
                Color.argb(alpha, red, green, 0)
            }
            else -> {
                Color.argb(alpha, 255, 0, 0) // Pure red for high intensity
            }
        }
    }
}