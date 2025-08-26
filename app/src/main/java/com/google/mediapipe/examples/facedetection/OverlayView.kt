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
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: FaceDetectorResult? = null
    private var boxPaint = Paint()
    private var eyePaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var fullBitmap: Bitmap? = null
    private var originalImageHeight: Int = 0
    private var originalImageWidth: Int = 0
    private var bounds = android.graphics.Rect()
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
    private val heatmapDecayTime = 8000L // 8 seconds - longer persistence for better visibility
    private val maxHeatmapValue = 100f
    private var dynamicContrastThreshold = 30.0
    
    // Enhanced contrast processing
    private var clahe: Any? = null
    private var contrastEnhancedFrames = HashMap<FaceRect, Mat>()
    private var adaptiveContrastHistory = mutableListOf<Double>()
    private val contrastHistorySize = 10
    private val minContrastThreshold = 5.0
    private val maxContrastThreshold = 80.0

    private val backgroundExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    private val ioScope = CoroutineScope(Dispatchers.IO)
    private var processingJob: Job? = null
    private var contrastJob: Job? = null
    private val lock = java.util.concurrent.locks.ReentrantLock()

    init {
        initPaints()
        System.loadLibrary("opencv_java4")
        initializeEyeCascade()
        initializeCLAHE()
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

    private fun initializeCLAHE() {
        try {
            // Use simple histogram equalization as fallback
            clahe = "histogram_equalization" // Simple marker for enhanced processing
        } catch (e: Exception) {
            Log.e("OverlayView", "Error initializing contrast enhancement: ${e.message}")
            clahe = null
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
            contrastEnhancedFrames.values.forEach { it.release() }
            contrastEnhancedFrames.clear()
            adaptiveContrastHistory.clear()
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
        
        // Always draw some debug info to verify drawing is working
        val debugPaint = Paint()
        debugPaint.color = Color.RED
        debugPaint.textSize = 30f
        debugPaint.style = Paint.Style.FILL
        canvas.drawText("OVERLAY ACTIVE", 50f, 50f, debugPaint)
        
        lock.lock()
        try {
            // Decay heatmap data
            decayHeatmap()
            
            results?.let {
                // Debug: Show number of detections
                val detectionCount = it.detections().size
                debugPaint.color = Color.YELLOW
                canvas.drawText("Faces: $detectionCount", 50f, 100f, debugPaint)
                
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
                    
                    val cachedBitmap = cachedFaceBitmaps[rectKey]
                    cachedBitmap?.let {
                        // Draw grayscale face bitmap with more transparency for heatmap visibility
                        val bitmapPaint = Paint()
                        bitmapPaint.alpha = 150 // More transparent to see heatmap better
                        canvas.drawBitmap(it, scaledLeft, scaledTop, bitmapPaint)
                    }
                    
                    // Draw heatmap on top for visibility
                    val heatmap = heatmapData[rectKey]
                    if (heatmap != null) {
                        drawHeatmap(canvas, rectKey, heatmap)
                        
                        // Debug: Draw a small indicator if heatmap has data
                        val maxValue = heatmap.maxOrNull() ?: 0f
                        val debugPaint = Paint()
                        debugPaint.color = Color.GREEN
                        debugPaint.textSize = 20f
                        debugPaint.style = Paint.Style.FILL
                        debugPaint.setShadowLayer(2f, 1f, 1f, Color.BLACK)
                        canvas.drawText("H:${maxValue.toInt()}", 
                            scaledLeft + 5, scaledTop + 25, debugPaint)
                    } else {
                        // Draw test pattern to ensure heatmap rendering works
                        drawTestHeatmap(canvas, rectKey)
                        
                        // Debug: Show that we're drawing test pattern
                        debugPaint.color = Color.RED
                        debugPaint.textSize = 16f
                        canvas.drawText("Drawing test pattern", scaledLeft, scaledTop + 60, debugPaint)
                    }
                    
                    // Optionally draw enhanced contrast frame (for debugging)
                    // drawEnhancedContrastFrame(canvas, rectKey)

                    // Draw the detection box with enhanced visibility
                    boxPaint.color = Color.GREEN
                    boxPaint.strokeWidth = 8F
                    boxPaint.style = Paint.Style.STROKE
                    canvas.drawRect(drawableRect, boxPaint)
                    
                    // Draw debug info about the face region
                    debugPaint.color = Color.CYAN
                    debugPaint.textSize = 16f
                    canvas.drawText("Face: ${boundingBox.width().toInt()}x${boundingBox.height().toInt()}", 
                        scaledLeft, scaledTop - 10, debugPaint)

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
            } ?: run {
                // Debug: Show if no results
                debugPaint.color = Color.WHITE
                canvas.drawText("No face detected", 50f, 100f, debugPaint)
            }
            
            // Debug: Show contrast detection status
            debugPaint.color = Color.MAGENTA
            debugPaint.textSize = 20f
            canvas.drawText("Heatmaps: ${heatmapData.size}", 50f, 150f, debugPaint)
            canvas.drawText("Face regions: ${lastFaceRegions.size}", 50f, 180f, debugPaint)
            
            // Debug: Show scaling and image info
            debugPaint.color = Color.CYAN
            debugPaint.textSize = 16f
            canvas.drawText("Scale: ${String.format("%.2f", uniformScaleFactor)}", 50f, 210f, debugPaint)
            canvas.drawText("Original: ${originalImageWidth}x${originalImageHeight}", 50f, 230f, debugPaint)
            canvas.drawText("Offset: ${xOffset.toInt()},${yOffset.toInt()}", 50f, 250f, debugPaint)
            
        } finally {
            lock.unlock()
        }
    }

    private fun applyTransformations(bitmap: Bitmap, drawableRect: RectF): Bitmap {
        val bitmapMatrix = Matrix()
        if (rotationDegrees != 0f) {
            bitmapMatrix.postRotate(rotationDegrees)
        }

        val scaleX = drawableRect.width() / bitmap.width.toFloat()
        val scaleY = drawableRect.height() / bitmap.height.toFloat()
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
    fun processContrastDetection(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled) return
        
        Log.d("OverlayView", "processContrastDetection called - bitmap: ${bitmap.width}x${bitmap.height}")
        
        contrastJob?.cancel()
        contrastJob = ioScope.launch {
            try {
                performContrastDetection(bitmap)
                withContext(Dispatchers.Main) {
                    if (!bitmap.isRecycled) {
                        invalidate()
                    }
                }
            } catch (e: Exception) {
                Log.e("OverlayView", "Error in contrast detection: ${e.message}")
            }
        }
    }
    
    private suspend fun performContrastDetection(currentBitmap: Bitmap) {
        if (currentBitmap.isRecycled) return
        
        lock.lock()
        try {
            if (lastFaceRegions.isEmpty()) return
            
            val currentMat = Mat()
            Utils.bitmapToMat(currentBitmap, currentMat)
            
            val grayCurrentMat = Mat()
            Imgproc.cvtColor(currentMat, grayCurrentMat, Imgproc.COLOR_RGB2GRAY)
            
            if (previousFrame != null) {
                for (faceRegion in lastFaceRegions) {
                    val faceRect = org.opencv.core.Rect(
                        max(0, faceRegion.left.toInt()),
                        max(0, faceRegion.top.toInt()),
                        min(currentBitmap.width, faceRegion.right.toInt()),
                        min(currentBitmap.height, faceRegion.bottom.toInt())
                    )
                    
                    if (faceRect.width > 0 && faceRect.height > 0) {
                        val faceKey = FaceRect(faceRect.x, faceRect.y, faceRect.x + faceRect.width, faceRect.y + faceRect.height)
                        
                        // Extract face region from current and previous frames
                        val currentFace = Mat(grayCurrentMat, faceRect as org.opencv.core.Rect)
                        val previousFace = Mat(previousFrame!!, faceRect as org.opencv.core.Rect)
                        
                        // Apply maximum contrast enhancement to both frames
                        val enhancedCurrentFace = enhanceContrastMax(currentFace)
                        val enhancedPreviousFace = enhanceContrastMax(previousFace)
                        
                        // Store enhanced frame for visualization (optional)
                        contrastEnhancedFrames[faceKey]?.release()
                        contrastEnhancedFrames[faceKey] = enhancedCurrentFace.clone()
                        
                        // Calculate frame difference on enhanced frames
                        val diff = Mat()
                        Core.absdiff(enhancedCurrentFace, enhancedPreviousFace, diff)
                        
                        // Apply dynamic adaptive threshold
                        val threshold = Mat()
                        val adaptiveThreshold = calculateAdaptiveDynamicThreshold(diff, faceKey)
                        Imgproc.threshold(diff, threshold, adaptiveThreshold, 255.0, Imgproc.THRESH_BINARY)
                        
                        // Apply morphological operations to reduce noise
                        val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
                        val cleanThreshold = Mat()
                        Imgproc.morphologyEx(threshold, cleanThreshold, Imgproc.MORPH_OPEN, morphKernel)
                        Imgproc.morphologyEx(cleanThreshold, cleanThreshold, Imgproc.MORPH_CLOSE, morphKernel)
                        
                        // Find contours with better parameters
                        val contours = mutableListOf<MatOfPoint>()
                        val hierarchy = Mat()
                        Imgproc.findContours(cleanThreshold, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
                        
                        // Filter contours by area to remove noise
                        val filteredContours = contours.filter { contour ->
                            val area = Imgproc.contourArea(contour)
                            area > 10.0 && area < (faceRect.width * faceRect.height * 0.1) // Between 10 pixels and 10% of face area
                        }
                        
                        // Update heatmap data with enhanced contours
                        updateHeatmapData(faceKey, filteredContours, faceRect)
                        
                        // Clean up
                        currentFace.release()
                        previousFace.release()
                        enhancedCurrentFace.release()
                        enhancedPreviousFace.release()
                        diff.release()
                        threshold.release()
                        cleanThreshold.release()
                        morphKernel.release()
                        hierarchy.release()
                        contours.forEach { it.release() }
                    }
                }
            }
            
            // Store current enhanced frame as previous for next iteration
            previousFrame?.release()
            previousFrame = grayCurrentMat.clone()
            
            currentMat.release()
            grayCurrentMat.release()
            
        } finally {
            lock.unlock()
        }
    }
    
    private fun enhanceContrastMax(inputMat: Mat): Mat {
        val enhanced = Mat()
        
        try {
            // Method 1: Histogram Equalization for enhanced contrast
            if (clahe != null) {
                Imgproc.equalizeHist(inputMat, enhanced)
            } else {
                inputMat.copyTo(enhanced)
            }
            
            // Method 2: Additional histogram stretching for maximum contrast
            val stretched = Mat()
            Core.normalize(enhanced, stretched, 0.0, 255.0, Core.NORM_MINMAX)
            
            // Method 3: Apply gamma correction for enhanced details
            val gamma = Mat()
            stretched.convertTo(gamma, -1, 1.2, 0.0) // Gamma = 1.2 for slight enhancement
            
            // Method 4: Apply unsharp masking for edge enhancement
            val blurred = Mat()
            val unsharpMask = Mat()
            Imgproc.GaussianBlur(gamma, blurred, Size(3.0, 3.0), 1.0)
            Core.addWeighted(gamma, 1.5, blurred, -0.5, 0.0, unsharpMask)
            
            // Clean up intermediate matrices
            stretched.release()
            gamma.release()
            blurred.release()
            
            return unsharpMask
            
        } catch (e: Exception) {
            Log.e("OverlayView", "Error enhancing contrast: ${e.message}")
            enhanced.release()
            return inputMat.clone()
        }
    }
    
    private fun calculateAdaptiveDynamicThreshold(diff: Mat, faceKey: FaceRect): Double {
        val mean = Core.mean(diff)
        val meanValue = mean.`val`[0]
        
        // Simplified approach: Calculate variance manually to avoid OpenCV type issues
        var sumSquaredDiff = 0.0
        var pixelCount = 0
        
        // Sample a subset of pixels for performance (every 4th pixel)
        for (row in 0 until diff.rows() step 4) {
            for (col in 0 until diff.cols() step 4) {
                val pixelValue = diff.get(row, col)[0]
                val diffFromMean = pixelValue - meanValue
                sumSquaredDiff += diffFromMean * diffFromMean
                pixelCount++
            }
        }
        
        val variance = if (pixelCount > 0) sumSquaredDiff / pixelCount else 0.0
        val stdDevValue = kotlin.math.sqrt(variance)
        
        // Add to adaptive history for this face region
        adaptiveContrastHistory.add(meanValue)
        if (adaptiveContrastHistory.size > contrastHistorySize) {
            adaptiveContrastHistory.removeAt(0)
        }
        
        // Calculate adaptive threshold based on:
        // 1. Current frame activity (mean + std dev)
        // 2. Historical activity (moving average)
        // 3. Enhanced contrast sensitivity
        
        val historicalMean = if (adaptiveContrastHistory.isNotEmpty()) {
            adaptiveContrastHistory.average()
        } else {
            meanValue
        }
        
        // Dynamic threshold calculation with maximum sensitivity for better heatmap
        val baseThreshold = when {
            meanValue < 5 -> minContrastThreshold * 0.5  // Very low activity - maximum sensitivity
            meanValue < 15 -> minContrastThreshold * 0.7 // Low activity - high sensitivity
            meanValue < 30 -> minContrastThreshold       // Medium activity - normal sensitivity
            meanValue < 50 -> minContrastThreshold * 1.2 // High activity - slightly less sensitive
            else -> maxContrastThreshold * 0.5           // Very high activity - reduce noise
        }
        
        // Adaptive component based on standard deviation (edge strength)
        val adaptiveComponent = stdDevValue * 0.3
        
        // Historical component to smooth out fluctuations
        val historicalComponent = (historicalMean - meanValue) * 0.2
        
        // Final adaptive threshold (explicit type conversion)
        val adaptiveThreshold = baseThreshold + adaptiveComponent + historicalComponent
        
        // Clamp to reasonable bounds
        dynamicContrastThreshold = adaptiveThreshold.coerceIn(minContrastThreshold, maxContrastThreshold)
        
        return dynamicContrastThreshold
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
    
    private fun updateHeatmapData(faceKey: FaceRect, contours: List<MatOfPoint>, faceRect: org.opencv.core.Rect) {
        val currentTime = System.currentTimeMillis()
        val width = faceRect.width
        val height = faceRect.height
        
        // Initialize heatmap array if it doesn't exist
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(width * height) { 0f }
            heatmapData[faceKey] = heatmap
        }
        
        // Add heat for each contour with enhanced intensity calculation
        for (contour in contours) {
            val points = contour.toArray()
            val contourArea = Imgproc.contourArea(contour)
            
            // Calculate heat intensity based on contour properties - increased for visibility
            val baseIntensity = when {
                contourArea < 20 -> 5f      // Small contours - increased from 2f
                contourArea < 50 -> 8f      // Medium contours - increased from 4f  
                contourArea < 100 -> 12f    // Large contours - increased from 7f
                else -> 15f                 // Very large contours - increased from 10f
            }
            
            // Apply heat with distance-based falloff for smoother heatmap
            for (point in points) {
                val centerX = point.x.toInt()
                val centerY = point.y.toInt()
                
                // Apply heat in a small radius around each contour point
                for (dy in -2..2) {
                    for (dx in -2..2) {
                        val x = centerX + dx
                        val y = centerY + dy
                        
                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            val index = (y * width) + x
                            if (index < heatmap.size) {
                                // Distance-based intensity falloff
                                val distance = sqrt((dx * dx + dy * dy).toDouble()).toFloat()
                                val intensity = baseIntensity * (1f / (1f + distance * 0.5f))
                                
                                heatmap[index] = min(maxHeatmapValue, heatmap[index] + intensity)
                            }
                        }
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
                // Decay existing heat values more slowly for better visibility
                val decayRate = 0.995f // Slower decay (was 0.98f)
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
        paint.isAntiAlias = true
        
        // Use smaller step size for better resolution and visibility
        for (y in 0 until height step 2) { // Changed from 4 to 2 for better resolution
            for (x in 0 until width step 2) { // Changed from 4 to 2 for better resolution
                val index = y * width + x
                if (index < heatmap.size) {
                    val intensity = heatmap[index] / maxHeatmapValue
                    if (intensity > 0.02f) { // Even lower threshold for maximum visibility
                        val color = getHeatmapColor(intensity)
                        paint.color = color
                        
                        val screenX = faceKey.left + x
                        val screenY = faceKey.top + y
                        canvas.drawRect(
                            (screenX * uniformScaleFactor) + xOffset,
                            (screenY * uniformScaleFactor) + yOffset,
                            ((screenX + 2) * uniformScaleFactor) + xOffset, // Match step size
                            ((screenY + 2) * uniformScaleFactor) + yOffset, // Match step size
                            paint
                        )
                    }
                }
            }
        }
    }
    
    private fun getHeatmapColor(intensity: Float): Int {
        // Enhanced color gradient with maximum visibility
        val alpha = (intensity * 200).toInt().coerceIn(80, 200) // Even more opaque, minimum 80 alpha
        
        return when {
            intensity < 0.2f -> {
                // Blue for low activity
                val blue = (255 * (intensity / 0.2f)).toInt().coerceIn(100, 255)
                Color.argb(alpha, 0, 0, blue)
            }
            intensity < 0.4f -> {
                // Cyan for low-medium activity
                val green = (255 * ((intensity - 0.2f) / 0.2f)).toInt()
                Color.argb(alpha, 0, green, 255)
            }
            intensity < 0.6f -> {
                // Green for medium activity
                val green = 255
                val blue = (255 * (1f - ((intensity - 0.4f) / 0.2f))).toInt()
                Color.argb(alpha, 0, green, blue)
            }
            intensity < 0.8f -> {
                // Yellow for medium-high activity
                val red = (255 * ((intensity - 0.6f) / 0.2f)).toInt()
                val green = 255
                Color.argb(alpha, red, green, 0)
            }
            else -> {
                // Red for high activity
                Color.argb(alpha, 255, 0, 0)
            }
        }
    }
    
    private fun drawTestHeatmap(canvas: Canvas, faceKey: FaceRect) {
        // Draw a simple test pattern to verify heatmap rendering works
        val paint = Paint()
        paint.style = Paint.Style.FILL
        paint.isAntiAlias = true
        
        val centerX = (faceKey.left + faceKey.right) / 2f
        val centerY = (faceKey.top + faceKey.bottom) / 2f
        
        // Draw a more visible gradient pattern
        for (radius in 20..60 step 10) {
            val intensity = (60 - radius) / 40f
            paint.color = getHeatmapColor(intensity)
            canvas.drawCircle(
                (centerX * uniformScaleFactor) + xOffset,
                (centerY * uniformScaleFactor) + yOffset,
                radius * uniformScaleFactor,
                paint
            )
        }
        
        // Draw some additional test squares
        for (i in 0..2) {
            paint.color = getHeatmapColor(0.3f + i * 0.3f)
            val size = 20f + i * 10f
            canvas.drawRect(
                (centerX * uniformScaleFactor) + xOffset - size/2 + i * 30f,
                (centerY * uniformScaleFactor) + yOffset + 40f,
                (centerX * uniformScaleFactor) + xOffset + size/2 + i * 30f,
                (centerY * uniformScaleFactor) + yOffset + 40f + size,
                paint
            )
        }
        
        // Test indicator - make it very visible
        val debugPaint = Paint()
        debugPaint.color = Color.YELLOW
        debugPaint.textSize = 24f
        debugPaint.style = Paint.Style.FILL
        debugPaint.setShadowLayer(3f, 2f, 2f, Color.BLACK)
        canvas.drawText("TEST HEATMAP", 
            (faceKey.left * uniformScaleFactor) + xOffset + 5, 
            (faceKey.top * uniformScaleFactor) + yOffset + 120, 
            debugPaint)
    }
    
    // Optional method to visualize enhanced contrast frames (for debugging)
    private fun drawEnhancedContrastFrame(canvas: Canvas, faceKey: FaceRect) {
        contrastEnhancedFrames[faceKey]?.let { enhancedMat ->
            try {
                val enhancedBitmap = Bitmap.createBitmap(
                    enhancedMat.cols(), 
                    enhancedMat.rows(), 
                    Bitmap.Config.ARGB_8888
                )
                Utils.matToBitmap(enhancedMat, enhancedBitmap)
                
                val paint = Paint()
                paint.alpha = 128 // Semi-transparent overlay
                
                canvas.drawBitmap(
                    enhancedBitmap,
                    (faceKey.left * uniformScaleFactor) + xOffset,
                    (faceKey.top * uniformScaleFactor) + yOffset,
                    paint
                )
                
                enhancedBitmap.recycle()
            } catch (e: Exception) {
                // Ignore visualization errors
            }
        }
    }
}