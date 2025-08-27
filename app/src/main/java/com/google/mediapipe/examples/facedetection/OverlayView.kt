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
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
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
    private var storedMediaPipePosition: RectF? = null // Store MediaPipe position for continuous contrast
    private var contrastBitmaps = HashMap<FaceRect, Bitmap>() // Store contrast detection results
    private val heatmapDecayTime = 8000L // 8 seconds - longer persistence for better visibility
    private val maxHeatmapValue = 100f
    private var dynamicContrastThreshold = 30.0
    
    // PYTHON-STYLE CONTRAST SYSTEM
    private var pythonHeatmap: Mat? = null // Python-style heatmap matrix
    private var heatmapMediPipePosition: RectF? = null // Store MediaPipe position for heatmap reference
    private val heatmapResetDistance = 100f // Reset heatmap if MediaPipe moves this far
    private var contrastFrameWidth = 0
    private var contrastFrameHeight = 0
    private val heatmapDecayInside = 0.98f // 2% decay inside face area (like Python)
    private val heatmapDecayOutside = 0.1f // 90% decay outside face area (like Python)
    private val heatIntensity = 0.4f // Heat intensity for new detections (like Python)
    private var pythonCurrentObjects = mutableListOf<FaceRect>() // Current detected objects
    
    // OLD PIXEL SYSTEM VARIABLES (Deprecated - keeping for compatibility)
    private var pixelContrastMap = FloatArray(0)
    private var pixelDecayTimestamps = LongArray(0)
    private val maxPixelIntensity = 1.0f
    private val pixelDecayRate = 0.98f
    private val pixelBoostAmount = 0.4f
    private val minVisibleIntensity = 0.15f
    private val timeBasedDecayRate = 50L
    
    // Enhanced contrast processing
    private var clahe: Any? = null
    private var contrastEnhancedFrames = HashMap<FaceRect, Mat>()
    private var adaptiveContrastHistory = mutableListOf<Double>()
    private val contrastHistorySize = 10
    private val minContrastThreshold = 5.0
    private val maxContrastThreshold = 80.0

    // Object tracking for consistent detection (Python version replication)
    private val objectHistory = ArrayDeque<List<FaceRect>>(150) // 150 frames history
    private val minConsistencyFrames = 10 // 10 frames for faster consistency
    private var currentObjects = mutableListOf<FaceRect>()
    private var lastConsistentFace: FaceRect? = null
    private var consistencyResetCounter = 0
    private val maxResetFrames = 40 // 40 frames for faster reset
    
    // Average face size tracking (dynamic exponential moving average)
    private var averageFaceSize: Float? = null
    private val faceAvgAlpha = 0.1f // 10% new, 90% old average
    
    // Movement tracking for coordinated detection
    private var previousMediaPipeCenterX: Float? = null
    private var previousMediaPipeCenterY: Float? = null
    private var previousFaceCenterX: Float? = null
    private var previousFaceCenterY: Float? = null

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

                    // Use original face coordinates for consistent key matching with contrast detection
                    val rectKey = FaceRect(
                        boundingBox.left.roundToInt(),
                        boundingBox.top.roundToInt(),
                        boundingBox.right.roundToInt(),
                        boundingBox.bottom.roundToInt()
                    )
                    
                    // Draw contrast detection result instead of grayscale
                    val contrastBitmap = contrastBitmaps[rectKey]
                    if (contrastBitmap != null) {
                        // Draw contrast bitmap (live contrast detection)
                        val bitmapPaint = Paint()
                        bitmapPaint.alpha = 180 // More visible contrast
                        canvas.drawBitmap(contrastBitmap, scaledLeft, scaledTop, bitmapPaint)
                        Log.d("OverlayView", "Drawing contrast bitmap at ${scaledLeft},${scaledTop}")
                    } else {
                        Log.d("OverlayView", "No contrast bitmap found for rectKey: $rectKey")
                        
                        // Draw a debug rectangle to show where contrast should be
                        val debugPaint = Paint().apply {
                            color = Color.GREEN
                            style = Paint.Style.STROKE
                            strokeWidth = 3f
                        }
                        canvas.drawRect(scaledLeft, scaledTop, scaledRight, scaledBottom, debugPaint)
                        
                        val debugTextPaint = Paint().apply {
                            color = Color.GREEN
                            textSize = 20f
                            isAntiAlias = true
                        }
                        canvas.drawText("NO CONTRAST", scaledLeft + 5, scaledTop + 25, debugTextPaint)
                    }
                    
                    // Draw heatmap on top for visibility
                    val heatmap = heatmapData[rectKey]
                    if (heatmap != null && heatmap.isNotEmpty()) {
                        val maxValue = heatmap.maxOrNull() ?: 0f
                        if (maxValue > 0.01f) { // Lower threshold for better visibility
                            drawHeatmap(canvas, rectKey, heatmap)
                            
                            // Debug: Draw a bright indicator if heatmap has data
                            val debugPaint = Paint()
                            debugPaint.color = Color.YELLOW
                            debugPaint.textSize = 24f
                            debugPaint.style = Paint.Style.FILL
                            debugPaint.setShadowLayer(3f, 2f, 2f, Color.BLACK)
                            canvas.drawText("HEAT:${String.format("%.1f", maxValue)}", 
                                scaledLeft + 5, scaledTop + 30, debugPaint)
                        } else {
                            // Heatmap exists but values too low
                            val debugPaint = Paint()
                            debugPaint.color = Color.CYAN
                            debugPaint.textSize = 18f
                            debugPaint.style = Paint.Style.FILL
                            debugPaint.setShadowLayer(2f, 1f, 1f, Color.BLACK)
                            canvas.drawText("LOW HEAT", scaledLeft + 5, scaledTop + 50, debugPaint)
                        }
                    } else {
                        // No heatmap data - show debug info
                        val debugPaint = Paint()
                        debugPaint.color = Color.RED
                        debugPaint.textSize = 18f
                        debugPaint.style = Paint.Style.FILL
                        debugPaint.setShadowLayer(2f, 1f, 1f, Color.BLACK)
                        canvas.drawText("NO MOTION DETECTED", scaledLeft + 5, scaledTop + 70, debugPaint)
                    }
                    
                    // Optionally draw enhanced contrast frame (for debugging)
                    // drawEnhancedContrastFrame(canvas, rectKey)

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
            
            // PYTHON HEATMAP DISPLAY: Draw Python-style colored heatmap overlay
            drawPythonHeatmapOverlay(canvas)
            
            // PYTHON STYLE DISPLAY: Draw Python-style detection results  
            drawPythonStyleResults(canvas)
            
        } finally {
            lock.unlock()
        }
    }
    
    private fun drawPythonStyleResults(canvas: Canvas) {
        // Draw MediaPipe face position (blue like Python Haar cascade)
        for (faceRegion in lastFaceRegions) {
            val faceRect = RectF(
                (faceRegion.left * uniformScaleFactor) + xOffset,
                (faceRegion.top * uniformScaleFactor) + yOffset,
                (faceRegion.right * uniformScaleFactor) + xOffset,
                (faceRegion.bottom * uniformScaleFactor) + yOffset
            )
            
            val mediaPipePaint = Paint().apply {
                color = Color.BLUE
                style = Paint.Style.STROKE
                strokeWidth = 3f
            }
            canvas.drawRect(faceRect, mediaPipePaint)
            
            val mediaPipeText = Paint().apply {
                color = Color.BLUE
                textSize = 24f
                isAntiAlias = true
                isFakeBoldText = true
            }
            canvas.drawText(
                "MediaPipe Face",
                faceRect.left,
                faceRect.top - 15,
                mediaPipeText
            )
            
            // Draw search area around MediaPipe face (yellow like Python)
            val searchMargin = 50f * uniformScaleFactor
            val searchPaint = Paint().apply {
                color = Color.YELLOW
                style = Paint.Style.STROKE
                strokeWidth = 1f
            }
            canvas.drawRect(
                faceRect.left - searchMargin,
                faceRect.top - searchMargin,
                faceRect.right + searchMargin,
                faceRect.bottom + searchMargin,
                searchPaint
            )
            
            val searchText = Paint().apply {
                color = Color.YELLOW
                textSize = 16f
                isAntiAlias = true
            }
            canvas.drawText(
                "Search Area",
                faceRect.left - searchMargin,
                faceRect.top - searchMargin - 5,
                searchText
            )
        }
        
        // OLD LINE-BASED DRAWING REMOVED - Using pixel-based contrast only
        
        // Draw the consistent face (green with "FACE" label like Python)
        lastConsistentFace?.let { consistentFace ->
            val consistentRect = RectF(
                (consistentFace.left * uniformScaleFactor) + xOffset,
                (consistentFace.top * uniformScaleFactor) + yOffset,
                (consistentFace.right * uniformScaleFactor) + xOffset,
                (consistentFace.bottom * uniformScaleFactor) + yOffset
            )
            
            val consistentPaint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 4f
            }
            canvas.drawRect(consistentRect, consistentPaint)
            
            val consistentText = Paint().apply {
                color = Color.GREEN
                textSize = 28f
                isAntiAlias = true
                isFakeBoldText = true
            }
            canvas.drawText(
                "FACE",
                consistentRect.left,
                consistentRect.top - 15,
                consistentText
            )
            
            // Draw face center point
            val centerX = consistentRect.left + (consistentRect.right - consistentRect.left) / 2
            val centerY = consistentRect.top + (consistentRect.bottom - consistentRect.top) / 2
            
            val centerPaint = Paint().apply {
                color = Color.RED
                style = Paint.Style.FILL
            }
            canvas.drawCircle(centerX, centerY, 6f, centerPaint)
        }
        
        // Draw detection info (Python style)
        val infoPaint = Paint().apply {
            color = Color.RED
            textSize = 28f
            isAntiAlias = true
            isFakeBoldText = true
            setShadowLayer(2f, 1f, 1f, Color.BLACK)
        }
        
        canvas.drawText("Face Features: PIXELS ONLY", 20f, 400f, infoPaint)
        canvas.drawText("Detection: INSIDE FACE", 20f, 440f, infoPaint)
        canvas.drawText("Reset Counter: $consistencyResetCounter", 20f, 480f, infoPaint)
        
        // Show Python-style performance info
        val perfPaint = Paint().apply {
            color = Color.CYAN
            textSize = 24f
            isAntiAlias = true
            setShadowLayer(1f, 1f, 1f, Color.BLACK)
        }
        canvas.drawText("OPTIMIZED CONTRAST DETECTION", 20f, 520f, perfPaint)
        canvas.drawText("MediaPipe Interval: 0.2s", 20f, 550f, perfPaint)
        canvas.drawText("Search Area Only - Max FPS", 20f, 580f, perfPaint)
        
        // Show contrast-based object count
        val contrastObjectCount = pythonCurrentObjects.size
        canvas.drawText("Contrast Objects: ${contrastObjectCount}", 20f, 610f, perfPaint)
        canvas.drawText("Stored MediaPipe Pos: ${if (storedMediaPipePosition != null) "YES" else "NO"}", 20f, 640f, perfPaint)
        canvas.drawText("Heatmap Active: ${if (pythonHeatmap != null) "YES" else "NO"}", 20f, 670f, perfPaint)
        
        // CONTRAST HEATMAP: Draw the heatmap overlay
        drawContrastHeatmapOverlay(canvas)
    }
    
    private fun drawContrastHeatmapOverlay(canvas: Canvas) {
        // Draw heatmap for contrast detection
        val facePosition = storedMediaPipePosition ?: lastFaceRegions.firstOrNull()
        
        if (facePosition != null) {
            val faceKey = FaceRect(
                facePosition.left.toInt(),
                facePosition.top.toInt(),
                facePosition.right.toInt(),
                facePosition.bottom.toInt()
            )
            
            val heatmap = heatmapData[faceKey]
            if (heatmap != null && heatmap.isNotEmpty()) {
                val maxHeat = heatmap.maxOrNull() ?: 0f
                if (maxHeat > 0.01f) {
                    // Draw heatmap with red color gradient
                    drawRedHeatmapOverlay(canvas, faceKey, heatmap)
                    
                    // Debug: Show heatmap status
                    val debugPaint = Paint().apply {
                        color = Color.WHITE
                        textSize = 24f
                        isAntiAlias = true
                        isFakeBoldText = true
                        setShadowLayer(2f, 1f, 1f, Color.BLACK)
                    }
                    canvas.drawText(
                        "CONTRAST HEATMAP: ${String.format("%.2f", maxHeat)}", 
                        20f, 610f, 
                        debugPaint
                    )
                } else {
                    // Show low heat debug
                    val debugPaint = Paint().apply {
                        color = Color.YELLOW
                        textSize = 20f
                        isAntiAlias = true
                        setShadowLayer(1f, 1f, 1f, Color.BLACK)
                    }
                    canvas.drawText("CONTRAST HEAT TOO LOW: ${String.format("%.4f", maxHeat)}", 20f, 610f, debugPaint)
                }
            } else {
                // Show no heatmap debug
                val debugPaint = Paint().apply {
                    color = Color.RED
                    textSize = 20f
                    isAntiAlias = true
                    setShadowLayer(1f, 1f, 1f, Color.BLACK)
                }
                canvas.drawText("NO CONTRAST HEATMAP DATA", 20f, 610f, debugPaint)
            }
        }
    }

    // OLD FUNCTION REMOVED - Using new Python-style heatmap implementation
    
    private fun drawRedHeatmapOverlay(canvas: Canvas, faceKey: FaceRect, heatmap: FloatArray) {
        // Get the actual camera frame dimensions for mapping
        val frameWidth = lastFaceRegions.firstOrNull()?.let { 
            (it.right - it.left) * 5 // Estimate frame width from face size
        } ?: 640f
        
        val frameHeight = lastFaceRegions.firstOrNull()?.let {
            (it.bottom - it.top) * 5 // Estimate frame height from face size  
        } ?: 480f
        
        // The heatmap is stored as frame_rows * frame_cols
        // We need to figure out the cols and rows from the camera frame
        val cols = frameWidth.toInt()
        val rows = frameHeight.toInt()
        
        // Draw heatmap pixels as red overlay
        val maxHeat = heatmap.maxOrNull() ?: 0f
        if (maxHeat <= 0f) {
            Log.d("OverlayView", "No heat to draw, maxHeat = $maxHeat")
            return
        }
        
        Log.d("OverlayView", "Drawing heatmap: ${cols}x${rows}, maxHeat=$maxHeat")
        
        // Use larger pixel size for visibility
        val pixelSize = 6f
        var pixelsDrawn = 0
        
        for (y in 0 until min(rows, heatmap.size / cols) step 3) { // Step 3 for performance
            for (x in 0 until cols step 3) {
                val index = y * cols + x
                if (index < heatmap.size) {
                    val intensity = heatmap[index] / maxHeat
                    
                    if (intensity > 0.005f) { // Very low threshold for visibility
                        // Convert frame coordinates to screen coordinates
                        val screenX = (x.toFloat() / cols.toFloat() * width * uniformScaleFactor) + xOffset
                        val screenY = (y.toFloat() / rows.toFloat() * height * uniformScaleFactor) + yOffset
                        
                        // Create bright red heatmap color (very visible)
                        val alpha = (intensity * 255 * 0.8f).toInt().coerceIn(100, 200) // More opaque
                        val red = (255 * intensity).toInt().coerceIn(150, 255) // Brighter red
                        val green = (50 * (1f - intensity)).toInt().coerceIn(0, 50) // Less green
                        val blue = 0
                        
                        val heatPaint = Paint().apply {
                            color = Color.argb(alpha, red, green, blue)
                            style = Paint.Style.FILL
                        }
                        
                        // Draw heat pixel
                        canvas.drawRect(
                            screenX, screenY,
                            screenX + pixelSize, screenY + pixelSize,
                            heatPaint
                        )
                        pixelsDrawn++
                    }
                }
            }
        }
        
        Log.d("OverlayView", "Drew $pixelsDrawn heat pixels")
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
            
            // Store MediaPipe position for continuous contrast processing
            if (lastFaceRegions.isNotEmpty()) {
                storedMediaPipePosition = lastFaceRegions.first() // Use first face
                Log.d("OverlayView", "Stored MediaPipe position for continuous contrast processing")
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
        // DISABLED: No more grayscale processing - contrast detection handles display
        Log.d("OverlayView", "Background processing disabled - using live contrast detection")
        return
        
        /*
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
                    boundingBox.left.roundToInt(),
                    boundingBox.top.roundToInt(),
                    boundingBox.right.roundToInt(),
                    boundingBox.bottom.roundToInt()
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
        */
    }
    
    // Process contrast detection for current frame
    fun processContrastDetection(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled) return
        
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
                // Ignore processing errors
            }
        }
    }
    
    private suspend fun performContrastDetection(currentBitmap: Bitmap) {
        if (currentBitmap.isRecycled) return
        
        lock.lock()
        try {
            if (lastFaceRegions.isEmpty()) return
            
            // Clear current objects for this frame
            currentObjects.clear()
            
            val currentMat = Mat()
            Utils.bitmapToMat(currentBitmap, currentMat)
            
            // LIVE CONTRAST DETECTION: Use stored MediaPipe position for continuous contrast processing
            performLiveContrastDetection(currentMat)
            
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
                        
                        // Use a much lower threshold to detect more motion
                        val finalThreshold = min(adaptiveThreshold, 5.0) // Very aggressive threshold for motion detection
                        Log.d("OverlayView", "Using threshold: $finalThreshold (adaptive was: $adaptiveThreshold)")
                        
                        Imgproc.threshold(diff, threshold, finalThreshold, 255.0, Imgproc.THRESH_BINARY)
                        
                        // Gentle morphological operations to reduce noise while preserving open contours
                        val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0, 2.0)) // Smaller kernel
                        val cleanThreshold = Mat()
                        
                        // Only apply opening to remove small noise, skip closing to preserve open contours
                        Imgproc.morphologyEx(threshold, cleanThreshold, Imgproc.MORPH_OPEN, morphKernel)
                        
                        // Find ALL contours including open/partial ones
                        val contours = mutableListOf<MatOfPoint>()
                        val hierarchy = Mat()
                        
                        // Use RETR_LIST to get all contours (including open ones) instead of RETR_EXTERNAL
                        // Use CHAIN_APPROX_NONE to preserve all contour points for better detection of partial shapes
                        Imgproc.findContours(cleanThreshold, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE)
                        
                        Log.d("OverlayView", "Found ${contours.size} raw contours for face ${faceRect.width}x${faceRect.height}")
                        
                        // Very permissive filtering to include open/partial contours
                        val filteredContours = contours.filter { contour ->
                            val area = Imgproc.contourArea(contour)
                            val points = contour.toArray()
                            
                            // Calculate arc length manually to avoid MatOfPoint2f compatibility issues
                            var arcLength = 0.0
                            for (i in 0 until points.size - 1) {
                                val p1 = points[i]
                                val p2 = points[i + 1]
                                val dx = p2.x - p1.x
                                val dy = p2.y - p1.y
                                arcLength += kotlin.math.sqrt(dx * dx + dy * dy)
                            }
                            
                            // Accept contours based on area OR arc length (for open contours)
                            val minArea = 1.0 // Ultra small minimum area
                            val maxArea = faceRect.width * faceRect.height * 0.5 // Allow even larger areas
                            val minArcLength = 5.0 // Lower minimum arc length for open contours
                            
                            (area > minArea && area < maxArea) || (arcLength > minArcLength)
                        }
                        
                        Log.d("OverlayView", "Filtered to ${filteredContours.size} contours (threshold: $adaptiveThreshold)")
                        
                        // Use real contour data for heat generation
                        if (filteredContours.isNotEmpty()) {
                            Log.d("OverlayView", "Creating real heat data from ${filteredContours.size} contours")
                            updateHeatmapData(faceKey, filteredContours, faceRect)
                        } else {
                            Log.d("OverlayView", "No contours found - trying direct diff heat generation")
                            // Generate heat directly from difference image if no contours found
                            createHeatFromDifference(faceKey, faceRect, diff)
                        }
                        
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
            
            // Update object tracking system (Python version logic)
            // FAST: Pixel-based system handles all visualization - no additional processing needed
            Log.d("OverlayView", "Pixel contrast processing complete")
            
        } finally {
            lock.unlock()
        }
    }
    
    private fun performLiveContrastDetection(currentMat: Mat) {
        // PYTHON-STYLE CONTRAST DETECTION: Replicate exact Python contrast detection logic
        pythonCurrentObjects.clear()
        
        // Use stored MediaPipe position (equivalent to Python's Haar cascade face)
        val mediaPipeFace = storedMediaPipePosition ?: lastFaceRegions.firstOrNull()
        
        Log.d("OverlayView", "=== PYTHON-STYLE CONTRAST DETECTION === Frame: ${currentMat.cols()}x${currentMat.rows()}")
        
        if (mediaPipeFace == null) {
            Log.d("OverlayView", "No MediaPipe face - using fallback center detection")
            
            // PYTHON FALLBACK: Use center of frame like Python does
            val centerX = currentMat.cols() / 2 - 100
            val centerY = currentMat.rows() / 2 - 100
            val defaultW = 200
            val defaultH = 200
            
            // Ensure within bounds
            val boundedX = maxOf(0, minOf(centerX, currentMat.cols() - defaultW))
            val boundedY = maxOf(0, minOf(centerY, currentMat.rows() - defaultH))
            
            pythonCurrentObjects.add(FaceRect(boundedX, boundedY, boundedX + defaultW, boundedY + defaultH))
            
            // Initialize heatmap for fallback
            initializePythonHeatmap(currentMat.cols(), currentMat.rows())
            updatePythonHeatmap(currentMat, mediaPipeFace)
            return
        }
        
        // PYTHON LOGIC: Use MediaPipe face as main detection, find contours inside face area
        performPythonContrastDetection(currentMat, mediaPipeFace)
        
        // Initialize and update Python-style heatmap
        initializePythonHeatmap(currentMat.cols(), currentMat.rows())
        updatePythonHeatmap(currentMat, mediaPipeFace)
        
        Log.d("OverlayView", "Python-style contrast detection complete")
    }
    
    private fun performPythonContrastDetection(currentMat: Mat, mediaPipeFace: RectF) {
        // OPTIMIZED: Contrast detection ONLY in search area for max FPS
        val faceX = maxOf(0, mediaPipeFace.left.toInt())
        val faceY = maxOf(0, mediaPipeFace.top.toInt())
        val faceW = minOf(currentMat.cols() - faceX, (mediaPipeFace.right - mediaPipeFace.left).toInt())
        val faceH = minOf(currentMat.rows() - faceY, (mediaPipeFace.bottom - mediaPipeFace.top).toInt())
        
        if (faceW <= 0 || faceH <= 0) return
        
        // OPTIMIZED: Define search area with minimal padding (5 pixels like Python)
        val searchPadding = 5
        val searchX = maxOf(0, faceX - searchPadding)
        val searchY = maxOf(0, faceY - searchPadding)
        val searchW = minOf(currentMat.cols() - searchX, faceW + 2 * searchPadding)
        val searchH = minOf(currentMat.rows() - searchY, faceH + 2 * searchPadding)
        
        // FAST: Extract ONLY search region for processing (not full frame)
        val searchRegion = Mat(currentMat, org.opencv.core.Rect(searchX, searchY, searchW, searchH))
        
        if (searchRegion.size().area() <= 0) {
            searchRegion.release()
            return
        }
        
        // CONTRAST DETECTION: Convert search area to grayscale for contrast analysis
        val graySearchRegion = Mat()
        Imgproc.cvtColor(searchRegion, graySearchRegion, Imgproc.COLOR_RGB2GRAY)
        
        // DYNAMIC CONTRAST: Apply contrast enhancement only to search area
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val enhancedSearchRegion = Mat()
        clahe.apply(graySearchRegion, enhancedSearchRegion)
        
        // FAST THRESHOLD: Calculate dynamic threshold for search area only
        val meanStdDev = MatOfDouble()
        val mean = MatOfDouble()
        Core.meanStdDev(enhancedSearchRegion, mean, meanStdDev)
        
        val meanArray = mean.toArray()
        val stdArray = meanStdDev.toArray()
        val searchMean = if (meanArray.isNotEmpty()) meanArray[0] else 128.0
        val searchStd = if (stdArray.isNotEmpty()) stdArray[0] else 20.0
        
        // CONTRAST-BASED THRESHOLD: Use contrast characteristics
        val contrastThreshold = searchMean + (searchStd * 0.5) // Adaptive threshold
        
        // CONTOUR DETECTION: Find contours in search area based on contrast
        val thresholdMat = Mat()
        Imgproc.threshold(enhancedSearchRegion, thresholdMat, contrastThreshold, 255.0, Imgproc.THRESH_BINARY)
        
        // FAST MORPHOLOGY: Clean up contours
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        val cleanMat = Mat()
        Imgproc.morphologyEx(thresholdMat, cleanMat, Imgproc.MORPH_OPEN, kernel)
        
        // FIND CONTOURS: Based on contrast in search area only
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(cleanMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        Log.d("OverlayView", "CONTRAST: Found ${contours.size} contours in search area (${searchW}x${searchH})")
        
        // FAST PROCESSING: Filter contours by size and aspect ratio for max FPS
        val mediaPipeSize = faceW * faceH
        val mediaPipeCenterX = faceX + faceW / 2
        val mediaPipeCenterY = faceY + faceH / 2
        
        val potentialObjects = mutableListOf<Array<Any>>() // [x, y, w, h, size, score]
        
        for (contour in contours) {
            val boundingRect = Imgproc.boundingRect(contour)
            val contourSize = boundingRect.width * boundingRect.height
            
            // FAST FILTERING: Relaxed size filtering for better detection
            val sizeRatio = contourSize.toFloat() / mediaPipeSize
            if (sizeRatio in 0.1f..5.0f) { // More permissive for contrast-based detection
                // FAST FILTERING: Relaxed aspect ratio
                val aspectRatio = boundingRect.width.toFloat() / boundingRect.height
                if (aspectRatio in 0.3f..3.0f) { // More permissive
                    // COORDINATES: Convert search area coordinates to full frame coordinates
                    val fullX = searchX + boundingRect.x
                    val fullY = searchY + boundingRect.y
                    val fullW = boundingRect.width
                    val fullH = boundingRect.height
                    
                    // SCORING: Distance-based scoring for best contour selection
                    val objCenterX = fullX + fullW / 2
                    val objCenterY = fullY + fullH / 2
                    val distance = kotlin.math.sqrt(
                        ((mediaPipeCenterX - objCenterX) * (mediaPipeCenterX - objCenterX) + 
                         (mediaPipeCenterY - objCenterY) * (mediaPipeCenterY - objCenterY)).toDouble()
                    ).toFloat()
                    
                    val sizeSimilarity = kotlin.math.abs(1.0f - (contourSize.toFloat() / mediaPipeSize))
                    val score = distance + (sizeSimilarity * 30f) // Reduced weight for faster processing
                    
                    potentialObjects.add(arrayOf(fullX, fullY, fullW, fullH, contourSize, score))
                }
            }
        }
        
        // FAST SELECTION: Find the best contrast-based object (lowest score)
        if (potentialObjects.isNotEmpty()) {
            val bestObject = potentialObjects.minByOrNull { it[5] as Float }
            if (bestObject != null) {
                val objX = bestObject[0] as Int
                val objY = bestObject[1] as Int
                val objW = bestObject[2] as Int
                val objH = bestObject[3] as Int
                
                pythonCurrentObjects.add(FaceRect(objX, objY, objX + objW, objY + objH))
                Log.d("OverlayView", "CONTRAST: Selected best object at ${objX},${objY} size ${objW}x${objH}")
            }
        } else {
            Log.d("OverlayView", "CONTRAST: No valid contours found, using MediaPipe face")
            // FALLBACK: Add MediaPipe face itself as object
            pythonCurrentObjects.add(FaceRect(faceX, faceY, faceX + faceW, faceY + faceH))
        }
        
        // FAST CLEANUP: Release only the matrices we created
        contours.forEach { it.release() }
        hierarchy.release()
        searchRegion.release()
        graySearchRegion.release()
        enhancedSearchRegion.release()
        meanStdDev.release()
        mean.release()
        thresholdMat.release()
        cleanMat.release()
        kernel.release()
        // Note: clahe object is automatically managed by OpenCV
    }
    
    private fun initializePythonHeatmap(width: Int, height: Int) {
        if (contrastFrameWidth != width || contrastFrameHeight != height) {
            contrastFrameWidth = width
            contrastFrameHeight = height
            
            pythonHeatmap?.release()
            pythonHeatmap = Mat.zeros(height, width, CvType.CV_32F)
            
            Log.d("OverlayView", "Initialized Python heatmap: ${width}x${height}")
        }
    }
    
    private fun updatePythonHeatmap(frameShape: Mat, mediaPipeFace: RectF?) {
        if (pythonHeatmap == null) return
        
        val currentMediaPipeCenter = if (mediaPipeFace != null) {
            val centerX = mediaPipeFace.left + (mediaPipeFace.right - mediaPipeFace.left) / 2
            val centerY = mediaPipeFace.top + (mediaPipeFace.bottom - mediaPipeFace.top) / 2
            Pair(centerX, centerY)
        } else null
        
        // PYTHON: Check if we need to reset heatmap due to MediaPipe movement
        if (heatmapMediPipePosition != null && mediaPipeFace != null) {
            val oldCenterX = heatmapMediPipePosition!!.left + (heatmapMediPipePosition!!.right - heatmapMediPipePosition!!.left) / 2
            val oldCenterY = heatmapMediPipePosition!!.top + (heatmapMediPipePosition!!.bottom - heatmapMediPipePosition!!.top) / 2
            
            val distance = kotlin.math.sqrt(
                ((currentMediaPipeCenter!!.first - oldCenterX) * (currentMediaPipeCenter.first - oldCenterX) + 
                 (currentMediaPipeCenter.second - oldCenterY) * (currentMediaPipeCenter.second - oldCenterY)).toDouble()
            ).toFloat()
            
            if (distance > heatmapResetDistance) {
                // Reset heatmap if MediaPipe moved too far
                pythonHeatmap = Mat.zeros(contrastFrameHeight, contrastFrameWidth, CvType.CV_32F)
                Log.d("OverlayView", "Python heatmap reset - MediaPipe moved ${distance.toInt()}px")
            }
        }
        
        // PYTHON: Apply decay with face area awareness (exact Python logic)
        if (mediaPipeFace != null) {
            val faceX = maxOf(0, mediaPipeFace.left.toInt())
            val faceY = maxOf(0, mediaPipeFace.top.toInt())
            val faceW = minOf(contrastFrameWidth - faceX, (mediaPipeFace.right - mediaPipeFace.left).toInt())
            val faceH = minOf(contrastFrameHeight - faceY, (mediaPipeFace.bottom - mediaPipeFace.top).toInt())
            
            // PYTHON: Add 5-pixel padding like Python
            val padding = 5
            val paddedX = maxOf(0, faceX - padding)
            val paddedY = maxOf(0, faceY - padding)
            val paddedW = minOf(contrastFrameWidth - paddedX, faceW + 2 * padding)
            val paddedH = minOf(contrastFrameHeight - paddedY, faceH + 2 * padding)
            
            // PYTHON: Apply different decay rates inside vs outside face
            val insideDecay = Scalar(heatmapDecayInside.toDouble()) // 2% decay inside
            val outsideDecay = Scalar(heatmapDecayOutside.toDouble()) // 90% decay outside
            
            // Create masks for inside and outside face area
            val insideMask = Mat.zeros(contrastFrameHeight, contrastFrameWidth, CvType.CV_8U)
            val insideRect = org.opencv.core.Rect(paddedX, paddedY, paddedW, paddedH)
            Imgproc.rectangle(insideMask, insideRect, Scalar(255.0), -1)
            
            val outsideMask = Mat.ones(contrastFrameHeight, contrastFrameWidth, CvType.CV_8U)
            Core.subtract(outsideMask, insideMask, outsideMask)
            
            // Apply decay
            val insideArea = Mat()
            val outsideArea = Mat()
            pythonHeatmap!!.copyTo(insideArea, insideMask)
            pythonHeatmap!!.copyTo(outsideArea, outsideMask)
            
            Core.multiply(insideArea, insideDecay, insideArea)
            Core.multiply(outsideArea, outsideDecay, outsideArea)
            
            Core.add(insideArea, outsideArea, pythonHeatmap)
            
            // Clean up
            insideMask.release()
            outsideMask.release()
            insideArea.release()
            outsideArea.release()
        } else {
            // No face - apply general decay
            val generalDecay = Scalar(heatmapDecayOutside.toDouble())
            Core.multiply(pythonHeatmap!!, generalDecay, pythonHeatmap!!)
        }
        
        // PYTHON: Add heat for current objects (exact Python intensity logic)
        for (obj in pythonCurrentObjects) {
            val objCenterX = obj.left + (obj.right - obj.left) / 2
            val objCenterY = obj.top + (obj.bottom - obj.top) / 2
            
            if (mediaPipeFace != null) {
                val faceCenterX = mediaPipeFace.left + (mediaPipeFace.right - mediaPipeFace.left) / 2
                val faceCenterY = mediaPipeFace.top + (mediaPipeFace.bottom - mediaPipeFace.top) / 2
                
                // PYTHON: Calculate distance-based heat intensity
                val distanceToCenter = kotlin.math.sqrt(
                    ((objCenterX - faceCenterX) * (objCenterX - faceCenterX) + 
                     (objCenterY - faceCenterY) * (objCenterY - faceCenterY)).toDouble()
                ).toFloat()
                
                val maxDistance = kotlin.math.sqrt(
                    ((mediaPipeFace.width() / 2 + 5) * (mediaPipeFace.width() / 2 + 5) + 
                     (mediaPipeFace.height() / 2 + 5) * (mediaPipeFace.height() / 2 + 5)).toDouble()
                ).toFloat()
                
                val distanceRatio = minOf(distanceToCenter / maxDistance, 1.0f)
                val heatIntensityFinal = heatIntensity * (1.0f - distanceRatio * 0.1f) // Python formula
                
                // PYTHON: Add heat in object area
                val objRect = org.opencv.core.Rect(
                    maxOf(0, obj.left),
                    maxOf(0, obj.top),
                    minOf(contrastFrameWidth - maxOf(0, obj.left), obj.right - obj.left),
                    minOf(contrastFrameHeight - maxOf(0, obj.top), obj.bottom - obj.top)
                )
                
                if (objRect.width > 0 && objRect.height > 0) {
                    val heatAddition = Mat(objRect.height, objRect.width, CvType.CV_32F, Scalar(heatIntensityFinal.toDouble()))
                    val roi = Mat(pythonHeatmap!!, objRect)
                    Core.add(roi, heatAddition, roi)
                    
                    heatAddition.release()
                    roi.release()
                }
            }
        }
        
        // Update stored MediaPipe position
        heatmapMediPipePosition = mediaPipeFace
        
        Log.d("OverlayView", "Python heatmap updated: ${pythonCurrentObjects.size} objects processed")
    }
    
    private fun drawPythonHeatmapOverlay(canvas: Canvas) {
        if (pythonHeatmap == null || contrastFrameWidth <= 0 || contrastFrameHeight <= 0) {
            return
        }
        
        try {
            // PYTHON: Normalize heatmap to 0-255 range (like Python cv2.applyColorMap)
            val minMaxLoc = Core.minMaxLoc(pythonHeatmap!!)
            val maxHeat = minMaxLoc.maxVal
            
            if (maxHeat > 0.0) {
                // Create normalized heatmap for color mapping
                val normalizedHeatmap = Mat()
                Core.normalize(pythonHeatmap!!, normalizedHeatmap, 0.0, 255.0, Core.NORM_MINMAX)
                normalizedHeatmap.convertTo(normalizedHeatmap, CvType.CV_8U)
                
                // GREEN HEAD DETECTION: Use green-based colormap for head detection
                val coloredHeatmap = Mat()
                // Use COLORMAP_SUMMER for green-yellow gradient (better for head detection)
                Imgproc.applyColorMap(normalizedHeatmap, coloredHeatmap, Imgproc.COLORMAP_SUMMER)
                
                // Convert to bitmap for Android drawing
                val heatmapBitmap = Bitmap.createBitmap(
                    coloredHeatmap.cols(), 
                    coloredHeatmap.rows(), 
                    Bitmap.Config.ARGB_8888
                )
                Utils.matToBitmap(coloredHeatmap, heatmapBitmap)
                
                // PYTHON: Blend with frame (30% opacity like Python cv2.addWeighted)
                val heatmapPaint = Paint().apply {
                    alpha = (0.3f * 255).toInt() // 30% opacity like Python
                }
                
                // Scale and position heatmap to match camera view
                val scaleX = uniformScaleFactor
                val scaleY = uniformScaleFactor
                
                val scaledHeatmap = Bitmap.createScaledBitmap(
                    heatmapBitmap,
                    (contrastFrameWidth * scaleX).toInt(),
                    (contrastFrameHeight * scaleY).toInt(),
                    true
                )
                
                // Draw heatmap overlay at correct position
                canvas.drawBitmap(scaledHeatmap, xOffset, yOffset, heatmapPaint)
                
                // Clean up
                normalizedHeatmap.release()
                coloredHeatmap.release()
                heatmapBitmap.recycle()
                scaledHeatmap.recycle()
                
                // Show heatmap status (like Python)
                val heatmapStatusPaint = Paint().apply {
                    color = Color.CYAN
                    textSize = 20f
                    isAntiAlias = true
                    setShadowLayer(2f, 1f, 1f, Color.BLACK)
                }
                canvas.drawText("Green Head Detection: ${String.format("%.3f", maxHeat)}", 20f, 700f, heatmapStatusPaint)
                canvas.drawText("Search Area: Contrast-based", 20f, 730f, heatmapStatusPaint)
                canvas.drawText("Max FPS Processing", 20f, 760f, heatmapStatusPaint)
                
                Log.d("OverlayView", "Drew Python heatmap overlay: max heat = ${String.format("%.3f", maxHeat)}")
            }
        } catch (e: Exception) {
            Log.e("OverlayView", "Error drawing Python heatmap: ${e.message}")
        }
    }

    private fun performPythonStyleDetection(currentMat: Mat) {
        // EXACT PYTHON REPLICATION: Use MediaPipe as main detection, then find closest contour with similar size
        currentObjects.clear()
        
        Log.d("OverlayView", "=== DETECTION START === Frame: ${currentMat.cols()}x${currentMat.rows()}")
        Log.d("OverlayView", "Face regions available: ${lastFaceRegions.size}")
        
        if (lastFaceRegions.isEmpty()) {
            Log.d("OverlayView", "No MediaPipe faces detected - using fallback")
            // Python: If no MediaPipe face, use a simple fallback (much faster)
            // Just use a default face area in the center of the frame
            val centerX = currentMat.cols() / 2 - 100
            val centerY = currentMat.rows() / 2 - 100
            val defaultW = 200
            val defaultH = 200
            
            // Ensure the default face area is within frame bounds
            val fallbackX = max(0, min(centerX, currentMat.cols() - defaultW))
            val fallbackY = max(0, min(centerY, currentMat.rows() - defaultH))
            
            currentObjects.add(FaceRect(fallbackX, fallbackY, fallbackX + defaultW, fallbackY + defaultH))
            Log.d("OverlayView", "Added fallback object: ${fallbackX},${fallbackY},${defaultW},${defaultH}")
            return
        }
        
        // Python: Use MediaPipe face as main detection reference (use first face)
        val mediaypipeFace = lastFaceRegions.first()
        val faceX = max(0, mediaypipeFace.left.toInt())
        val faceY = max(0, mediaypipeFace.top.toInt()) 
        val faceW = min(currentMat.cols() - faceX, (mediaypipeFace.right - mediaypipeFace.left).toInt())
        val faceH = min(currentMat.rows() - faceY, (mediaypipeFace.bottom - mediaypipeFace.top).toInt())
        val faceSize = faceW * faceH
        val faceCenterX = faceX + faceW / 2
        val faceCenterY = faceY + faceH / 2
        
        // Python: Define dynamic search area: MediaPipe face + 5 pixel padding  
        val searchPadding = 5
        val searchX = max(0, faceX - searchPadding)
        val searchY = max(0, faceY - searchPadding)
        val searchW = min(currentMat.cols() - searchX, faceW + 2 * searchPadding)
        val searchH = min(currentMat.rows() - searchY, faceH + 2 * searchPadding)
        
        if (searchW <= 0 || searchH <= 0) return
        
        // Extract search region (much smaller area for speed)
        val searchRegion = Mat(currentMat, org.opencv.core.Rect(searchX, searchY, searchW, searchH))
        
        // Python: Convert to HSV for skin tone detection
        val hsvRegion = Mat()
        Imgproc.cvtColor(searchRegion, hsvRegion, Imgproc.COLOR_RGB2HSV)
        
        // Python: Optimized skin tone range for face detection (made more permissive)
        val lowerSkin = Scalar(0.0, 20.0, 40.0, 0.0)  // Lower saturation and value thresholds (4 params)
        val upperSkin = Scalar(25.0, 255.0, 255.0, 255.0)  // Wider hue range (4 params)
        
        // Create a mask for skin tone
        val mask = Mat()
        Core.inRange(hsvRegion, lowerSkin, upperSkin, mask)
        
        // Python: Fast morphological operations
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        val erodedMask = Mat()
        val dilatedMask = Mat()
        Imgproc.erode(mask, erodedMask, kernel, Point(-1.0, -1.0), 1)
        Imgproc.dilate(erodedMask, dilatedMask, kernel, Point(-1.0, -1.0), 1)
        
        // Find contours in the mask
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(dilatedMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        Log.d("OverlayView", "MediaPipe face: ${faceX},${faceY},${faceW},${faceH} (size=${faceSize})")
        Log.d("OverlayView", "Search area: ${searchX},${searchY},${searchW},${searchH}")
        Log.d("OverlayView", "Found ${contours.size} contours in HSV skin mask")
        
        // PYTHON LOGIC: Find the closest contour with similar size to MediaPipe face
        val potentialObjects = mutableListOf<Pair<FaceRect, Double>>() // <object, score>
        
        for ((index, contour) in contours.withIndex()) {
            // Get bounding rectangle
            val boundingRect = Imgproc.boundingRect(contour)
            val contourSize = boundingRect.width * boundingRect.height
            
            // Python: Filter by reasonable size (similar to MediaPipe face)
            val sizeRatio = contourSize.toFloat() / faceSize.toFloat()
            Log.d("OverlayView", "Contour $index: ${boundingRect.width}x${boundingRect.height} (size=$contourSize, ratio=${String.format("%.2f", sizeRatio)})")
            
            if (sizeRatio in 0.1f..5.0f) {  // Much more permissive: 10% to 500% of MediaPipe size
                // Python: Filter for face-like aspect ratios (more permissive)
                val aspectRatio = boundingRect.width.toFloat() / boundingRect.height.toFloat()
                Log.d("OverlayView", "  Size passed, aspect ratio: ${String.format("%.2f", aspectRatio)}")
                
                if (aspectRatio in 0.2f..5.0f) {  // Much more permissive aspect ratios
                    // Convert coordinates back to full frame
                    val fullX = searchX + boundingRect.x
                    val fullY = searchY + boundingRect.y
                    val fullW = boundingRect.width
                    val fullH = boundingRect.height
                    
                    val objCenterX = fullX + fullW / 2
                    val objCenterY = fullY + fullH / 2
                    
                    // Python: Calculate distance from MediaPipe face center
                    val distance = kotlin.math.sqrt(
                        ((faceCenterX - objCenterX) * (faceCenterX - objCenterX) + 
                         (faceCenterY - objCenterY) * (faceCenterY - objCenterY)).toDouble()
                    )
                    
                    // Python: Calculate size similarity (closer to 1.0 is better)
                    val sizeSimilarity = kotlin.math.abs(1.0 - sizeRatio)
                    
                    // Python: Combined score: distance + size similarity (weighted)
                    val score = distance + (sizeSimilarity * 50.0)  // Weight size similarity
                    
                    potentialObjects.add(Pair(
                        FaceRect(fullX, fullY, fullX + fullW, fullY + fullH),
                        score
                    ))
                }
            }
        }
        
        // Python: Find the best object (lowest score = closest + most similar size)
        Log.d("OverlayView", "Potential objects after filtering: ${potentialObjects.size}")
        
        if (potentialObjects.isNotEmpty()) {
            val bestMatch = potentialObjects.minByOrNull { it.second }
            val bestObject = bestMatch?.first
            val bestScore = bestMatch?.second
            
            if (bestObject != null) {
                currentObjects.add(bestObject)  // PYTHON: Only ONE object per frame
                Log.d("OverlayView", "Selected best object: score=${String.format("%.1f", bestScore)}, rect=${bestObject.left},${bestObject.top},${bestObject.right-bestObject.left},${bestObject.bottom-bestObject.top}")
            }
        } else {
            Log.d("OverlayView", "No objects passed filtering - no skin contours found!")
            
            // FALLBACK: Create a test object in the center of MediaPipe face for debugging
            val testX = searchX + searchW / 4
            val testY = searchY + searchH / 4  
            val testW = searchW / 2
            val testH = searchH / 2
            val testObject = FaceRect(testX, testY, testX + testW, testY + testH)
            currentObjects.add(testObject)
            Log.d("OverlayView", "Added fallback test object for debugging: ${testX},${testY},${testW},${testH}")
        }
        
        // Clean up
        contours.forEach { it.release() }
        hierarchy.release()
        searchRegion.release()
        hsvRegion.release()
        mask.release()
        erodedMask.release()
        dilatedMask.release()
        kernel.release()
    }

    // OLD PIXEL SYSTEM REMOVED - Using Python-style heatmap now
    
    private fun updatePixelContrastFromContours(contours: List<MatOfPoint>, faceX: Int, faceY: Int, faceW: Int, faceH: Int) {
        // OLD PIXEL SYSTEM - FUNCTION DISABLED
        return
        
        val currentTime = System.currentTimeMillis()
        val faceStartIndex = 0 // We'll work within the full frame
        
        // First, apply decay to all pixels
        for (i in pixelContrastMap.indices) {
            val timeSinceUpdate = currentTime - pixelDecayTimestamps[i]
            
            if (timeSinceUpdate > timeBasedDecayRate) {
                // Time-based decay for old pixels
                pixelContrastMap[i] *= 0.95f // Faster decay for old pixels
            } else {
                // Frame-based decay for recent pixels
                pixelContrastMap[i] *= pixelDecayRate
            }
            
            // Remove very dim pixels
            if (pixelContrastMap[i] < minVisibleIntensity * 0.5f) {
                pixelContrastMap[i] = 0f
            }
        }
        
        // Add intensity from new contours
        var pixelsUpdated = 0
        for (contour in contours) {
            val points = contour.toArray()
            
            for (point in points) {
                // Convert contour point to full frame coordinates
                val fullX = (faceX + point.x).toInt()
                val fullY = (faceY + point.y).toInt()
                
                // Check bounds
                if (fullX >= 0 && fullX < contrastFrameWidth && fullY >= 0 && fullY < contrastFrameHeight) {
                    val pixelIndex = fullY * contrastFrameWidth + fullX
                    
                    if (pixelIndex >= 0 && pixelIndex < pixelContrastMap.size) {
                        // Boost pixel intensity (accumulation effect)
                        val currentIntensity = pixelContrastMap[pixelIndex]
                        val newIntensity = minOf(maxPixelIntensity, currentIntensity + pixelBoostAmount)
                        
                        pixelContrastMap[pixelIndex] = newIntensity
                        pixelDecayTimestamps[pixelIndex] = currentTime
                        pixelsUpdated++
                        
                        // Also boost neighboring pixels for visibility
                        for (dx in -1..1) {
                            for (dy in -1..1) {
                                val neighborX = fullX + dx
                                val neighborY = fullY + dy
                                
                                if (neighborX >= 0 && neighborX < contrastFrameWidth && 
                                    neighborY >= 0 && neighborY < contrastFrameHeight) {
                                    val neighborIndex = neighborY * contrastFrameWidth + neighborX
                                    
                                    if (neighborIndex >= 0 && neighborIndex < pixelContrastMap.size) {
                                        val neighborBoost = pixelBoostAmount * 0.5f // Half intensity for neighbors
                                        val neighborCurrent = pixelContrastMap[neighborIndex]
                                        pixelContrastMap[neighborIndex] = minOf(maxPixelIntensity, neighborCurrent + neighborBoost)
                                        pixelDecayTimestamps[neighborIndex] = currentTime
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Log.d("OverlayView", "Updated ${pixelsUpdated} pixels from ${contours.size} contours")
    }
    
    private fun drawPixelContrast(canvas: Canvas) {
        // OLD PIXEL SYSTEM - FUNCTION DISABLED
        return
        
        val paint = Paint().apply {
            style = Paint.Style.FILL
        }
        
        var visiblePixels = 0
        val pixelSize = 3f // Larger pixels for better visibility
        
        // Use the same scale factors as MediaPipe face detection for proper alignment
        val scaleX = uniformScaleFactor
        val scaleY = uniformScaleFactor
        
        for (y in 0 until contrastFrameHeight step 2) { // Skip every other pixel for performance
            for (x in 0 until contrastFrameWidth step 2) {
                val pixelIndex = y * contrastFrameWidth + x
                
                if (pixelIndex < pixelContrastMap.size) {
                    val intensity = pixelContrastMap[pixelIndex]
                    
                    if (intensity >= minVisibleIntensity) {
                        // Map intensity to color (red with varying alpha and brightness)
                        val alpha = (intensity * 255).toInt().coerceIn(0, 255)
                        val brightness = (intensity * 255).toInt().coerceIn(100, 255)
                        
                        paint.color = Color.argb(alpha, brightness, 0, 0) // Red with intensity-based alpha/brightness
                        
                        // Draw pixel as small rectangle with proper alignment
                        val screenX = (x * scaleX) + xOffset
                        val screenY = (y * scaleY) + yOffset
                        
                        canvas.drawRect(
                            screenX, 
                            screenY, 
                            screenX + pixelSize, 
                            screenY + pixelSize, 
                            paint
                        )
                        visiblePixels++
                    }
                }
            }
        }
        
        Log.d("OverlayView", "Drew ${visiblePixels} visible contrast pixels")
    }
    
    private fun updateContrastBasedHeatmap(frameShape: Mat) {
        // CONTRAST HEATMAP: Generate heatmap from contrast contours (not HSV skin)
        val facePosition = storedMediaPipePosition ?: lastFaceRegions.firstOrNull()
        
        Log.d("OverlayView", "=== CONTRAST HEATMAP === Objects: ${currentObjects.size}")
        
        if (facePosition == null) {
            Log.d("OverlayView", "No face position for contrast heatmap")
            return
        }
        
        val faceX = facePosition.left.toInt()
        val faceY = facePosition.top.toInt()
        val faceW = (facePosition.right - facePosition.left).toInt()
        val faceH = (facePosition.bottom - facePosition.top).toInt()
        val faceCenterX = faceX + faceW / 2
        val faceCenterY = faceY + faceH / 2
        
        // Initialize heatmap if needed
        val faceKey = FaceRect(faceX, faceY, faceX + faceW, faceY + faceH)
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(frameShape.rows() * frameShape.cols()) { 0f }
            heatmapData[faceKey] = heatmap
            Log.d("OverlayView", "Created contrast heatmap: ${frameShape.cols()}x${frameShape.rows()} = ${heatmap.size} pixels")
        }
        
        // Apply decay to all heatmap areas
        for (y in 0 until frameShape.rows()) {
            for (x in 0 until frameShape.cols()) {
                val index = y * frameShape.cols() + x
                if (index < heatmap.size) {
                    // Check if pixel is inside face area
                    val isInsideFace = (x >= faceX && x < faceX + faceW && 
                                       y >= faceY && y < faceY + faceH)
                    
                    if (isInsideFace) {
                        // Slow decay inside face area
                        heatmap[index] *= 0.98f
                    } else {
                        // Fast decay outside face area
                        heatmap[index] *= 0.9f
                    }
                }
            }
        }
        
        // Add heat for contrast objects (much more aggressive than HSV)
        for (obj in currentObjects) {
            val objCenterX = obj.left + (obj.right - obj.left) / 2
            val objCenterY = obj.top + (obj.bottom - obj.top) / 2
            
            // Calculate distance from object center to face center
            val distanceToCenter = kotlin.math.sqrt(
                ((objCenterX - faceCenterX) * (objCenterX - faceCenterX) + 
                 (objCenterY - faceCenterY) * (objCenterY - faceCenterY)).toDouble()
            ).toFloat()
            
            // Much higher heat intensity for contrast detection (easier to see)
            val maxDistance = kotlin.math.sqrt((faceW * faceW + faceH * faceH).toDouble()).toFloat()
            val distanceRatio = min(distanceToCenter / maxDistance, 1.0f)
            val heatIntensity = 2.0f * (1.0f - distanceRatio * 0.2f) // Much higher base intensity
            
            // Add heat in larger area around the object (more visible)
            var heatAdded = 0
            for (y in (obj.top - 2) until (obj.bottom + 2)) {
                for (x in (obj.left - 2) until (obj.right + 2)) {
                    if (y >= 0 && y < frameShape.rows() && x >= 0 && x < frameShape.cols()) {
                        val index = y * frameShape.cols() + x
                        if (index < heatmap.size) {
                            heatmap[index] = min(maxHeatmapValue, heatmap[index] + heatIntensity)
                            heatAdded++
                        }
                    }
                }
            }
            Log.d("OverlayView", "Added contrast heat: intensity=${String.format("%.1f", heatIntensity)} to $heatAdded pixels")
        }
        
        heatmapAge[faceKey] = System.currentTimeMillis()
    }

    private fun updatePythonStyleHeatmap(frameShape: Mat) {
        // PYTHON HEATMAP: Exact replication of Python heatmap logic
        val facePosition = lastFaceRegions.firstOrNull() // Use first face like Python
        
        Log.d("OverlayView", "=== HEATMAP UPDATE === Objects: ${currentObjects.size}")
        
        if (facePosition == null) {
            Log.d("OverlayView", "No face position for heatmap")
            return
        }
        
        val faceX = facePosition.left.toInt()
        val faceY = facePosition.top.toInt()
        val faceW = (facePosition.right - facePosition.left).toInt()
        val faceH = (facePosition.bottom - facePosition.top).toInt()
        val faceCenterX = faceX + faceW / 2
        val faceCenterY = faceY + faceH / 2
        
        // Initialize heatmap if needed
        val faceKey = FaceRect(faceX, faceY, faceX + faceW, faceY + faceH)
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(frameShape.rows() * frameShape.cols()) { 0f }
            heatmapData[faceKey] = heatmap
            Log.d("OverlayView", "Created heatmap: ${frameShape.cols()}x${frameShape.rows()} = ${heatmap.size} pixels")
        }
        
        // Python: Add minimal padding to detection area (5 pixels around face)
        val padding = 5
        val paddedX = max(0, faceX - padding)
        val paddedY = max(0, faceY - padding)
        val paddedW = min(frameShape.cols() - paddedX, faceW + 2 * padding)
        val paddedH = min(frameShape.rows() - paddedY, faceH + 2 * padding)
        
        // PYTHON: FAST HEATMAP CLEANUP - remove heatmap outside face area
        for (y in 0 until frameShape.rows()) {
            for (x in 0 until frameShape.cols()) {
                val index = y * frameShape.cols() + x
                if (index < heatmap.size) {
                    // Check if pixel is inside padded face area
                    val isInsideFace = (x >= paddedX && x < paddedX + paddedW && 
                                       y >= paddedY && y < paddedY + paddedH)
                    
                    if (isInsideFace) {
                        // Python: Apply normal decay to areas inside face (2% decay)
                        heatmap[index] *= 0.98f
                    } else {
                        // Python: Apply aggressive decay to areas outside face (90% decay per frame)
                        heatmap[index] *= 0.1f
                    }
                }
            }
        }
        
        // Python: Add heat for current objects inside padded area
        for (obj in currentObjects) {
            val objCenterX = obj.left + (obj.right - obj.left) / 2
            val objCenterY = obj.top + (obj.bottom - obj.top) / 2
            
            // Check if object center is inside padded area
            if (objCenterX >= paddedX && objCenterX < paddedX + paddedW &&
                objCenterY >= paddedY && objCenterY < paddedY + paddedH) {
                
                // Python: Calculate distance from object center to face center
                val distanceToCenter = kotlin.math.sqrt(
                    ((objCenterX - faceCenterX) * (objCenterX - faceCenterX) + 
                     (objCenterY - faceCenterY) * (objCenterY - faceCenterY)).toDouble()
                ).toFloat()
                
                // Python: Simplified heat intensity based on distance
                val maxDistance = kotlin.math.sqrt(
                    ((faceW/2 + padding) * (faceW/2 + padding) + 
                     (faceH/2 + padding) * (faceH/2 + padding)).toDouble()
                ).toFloat()
                val distanceRatio = min(distanceToCenter / maxDistance, 1.0f)
                val heatIntensity = 0.4f * (1.0f - distanceRatio * 0.1f) // Higher intensity for longer heatmap life
                
                // Add heat in the object area
                var heatAdded = 0
                for (y in obj.top until obj.bottom) {
                    for (x in obj.left until obj.right) {
                        if (y >= 0 && y < frameShape.rows() && x >= 0 && x < frameShape.cols()) {
                            val index = y * frameShape.cols() + x
                            if (index < heatmap.size) {
                                heatmap[index] = min(maxHeatmapValue, heatmap[index] + heatIntensity)
                                heatAdded++
                            }
                        }
                    }
                }
                Log.d("OverlayView", "Added heat: intensity=$heatIntensity to $heatAdded pixels, obj=${obj.left},${obj.top},${obj.right-obj.left},${obj.bottom-obj.top}")
            }
        }
        
        heatmapAge[faceKey] = System.currentTimeMillis()
    }

    private fun enhanceContrastForDetection(grayMat: Mat): Mat {
        // Apply maximum contrast enhancement for motion detection
        val enhanced = Mat()
        
        // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        val clahe = Imgproc.createCLAHE(3.0, Size(8.0, 8.0))
        clahe.apply(grayMat, enhanced)
        
        // Additional contrast stretching
        val stretched = Mat()
        Core.normalize(enhanced, stretched, 0.0, 255.0, Core.NORM_MINMAX)
        
        enhanced.release()
        return stretched
    }
    
    private fun calculateDynamicThreshold(enhancedMat: Mat): Double {
        // Calculate image statistics for dynamic thresholding
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(enhancedMat, mean, stddev)
        
        val meanArray = mean.toArray()
        val stdArray = stddev.toArray()
        val meanValue = if (meanArray.isNotEmpty()) meanArray[0] else 0.0
        val stdValue = if (stdArray.isNotEmpty()) stdArray[0] else 0.0
        
        // Adaptive threshold based on image statistics
        var threshold = meanValue + (stdValue * 0.5) // Base threshold
        
        // Add to contrast history for temporal smoothing
        adaptiveContrastHistory.add(threshold)
        if (adaptiveContrastHistory.size > contrastHistorySize) {
            adaptiveContrastHistory.removeAt(0)
        }
        
        // Calculate smoothed threshold
        if (adaptiveContrastHistory.isNotEmpty()) {
            threshold = adaptiveContrastHistory.average()
        }
        
        // Clamp to reasonable range for motion detection
        threshold = threshold.coerceIn(10.0, 80.0)
        
        Log.d("OverlayView", "Dynamic threshold: mean=${String.format("%.1f", meanValue)}, std=${String.format("%.1f", stdValue)}, final=${String.format("%.1f", threshold)}")
        
        return threshold
    }
    
    private fun createContrastBitmap(contrastMat: Mat): Bitmap? {
        return try {
            // Convert to RGB for better display
            val rgbMat = Mat()
            Imgproc.cvtColor(contrastMat, rgbMat, Imgproc.COLOR_GRAY2RGB)
            
            val bitmap = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgbMat, bitmap)
            
            rgbMat.release()
            Log.d("OverlayView", "Created contrast bitmap: ${bitmap.width}x${bitmap.height}")
            bitmap
        } catch (e: Exception) {
            Log.e("OverlayView", "Error creating contrast bitmap: ${e.message}")
            null
        }
    }
    
    private fun scaleContrastBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap? {
        return try {
            // Scale to match the face region size for proper display
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
            Log.d("OverlayView", "Scaled contrast bitmap: ${bitmap.width}x${bitmap.height} -> ${scaledBitmap.width}x${scaledBitmap.height}")
            scaledBitmap
        } catch (e: Exception) {
            Log.e("OverlayView", "Error scaling contrast bitmap: ${e.message}")
            null
        }
    }

    private fun updateObjectTracking() {
        // Check if consistency should be reset
        if (shouldResetConsistency()) {
            objectHistory.clear()
            lastConsistentFace = null
            Log.d("OverlayView", "Consistency reset - no face detected for too long")
        }
        
        // Add current objects to history
        objectHistory.addLast(currentObjects.toList())
        if (objectHistory.size > 150) {
            objectHistory.removeFirst()
        }
        
        // Find the most consistent object
        val consistentFace = findMostConsistentObject()
        
        // Update consistent face position if found
        if (consistentFace != null) {
            lastConsistentFace = updateConsistentFacePosition(consistentFace, consistentFace)
        } else if (lastConsistentFace != null && currentObjects.isNotEmpty()) {
            // Try to reconnect to nearest object
            var nearestObject: FaceRect? = null
            var nearestDistance = Float.MAX_VALUE
            
            for (obj in currentObjects) {
                val distance = calculateDistance(lastConsistentFace!!, obj)
                if (distance < nearestDistance) {
                    nearestDistance = distance
                    nearestObject = obj
                }
            }
            
            // Reconnect if close enough (150px threshold like Python version)
            if (nearestObject != null && nearestDistance < 150f) {
                lastConsistentFace = nearestObject
                Log.d("OverlayView", "Reconnecting lost face to nearest object at distance ${nearestDistance.toInt()}px")
            }
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
        
        // Much more aggressive dynamic threshold for better motion detection
        val baseThreshold = when {
            meanValue < 3 -> minContrastThreshold * 0.2   // Very low activity - ultra sensitive
            meanValue < 8 -> minContrastThreshold * 0.4   // Low activity - super sensitive
            meanValue < 20 -> minContrastThreshold * 0.6  // Medium activity - high sensitivity
            meanValue < 40 -> minContrastThreshold * 0.8  // Higher activity - still sensitive
            else -> minContrastThreshold * 1.0            // High activity - normal sensitivity
        }
        
        // Boost adaptive component for better edge detection
        val adaptiveComponent = stdDevValue * 0.5 // Increased multiplier
        
        // More responsive to temporal changes
        val historicalComponent = (historicalMean - meanValue) * 0.4 // Doubled responsiveness
        
        // Add motion boost for any significant change
        val motionBoost = if (kotlin.math.abs(meanValue - historicalMean) > 2.0) {
            kotlin.math.abs(meanValue - historicalMean) * 0.2
        } else 0.0
        
        // Final adaptive threshold with motion enhancement
        val adaptiveThreshold = baseThreshold + adaptiveComponent + historicalComponent - motionBoost
        
        // More permissive bounds for better detection
        dynamicContrastThreshold = adaptiveThreshold.coerceIn(minContrastThreshold * 0.1, maxContrastThreshold * 0.7)
        
        return dynamicContrastThreshold
    }

    
    private fun updateHeatmapData(faceKey: FaceRect, contours: List<MatOfPoint>, faceRect: org.opencv.core.Rect) {
        val currentTime = System.currentTimeMillis()
        val width = faceRect.width
        val height = faceRect.height
        
        // Debug: Log contour detection
        if (contours.isNotEmpty()) {
            Log.d("OverlayView", "Updating heatmap with ${contours.size} contours for face ${width}x${height}")
        }
        
        // Initialize heatmap array if it doesn't exist
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(width * height) { 0f }
            heatmapData[faceKey] = heatmap
            Log.d("OverlayView", "Created new heatmap array: ${width}x${height} = ${heatmap.size} pixels")
        }
        
        // Add heat for each contour with enhanced intensity for open/partial contours
        for (contour in contours) {
            val points = contour.toArray()
            val contourArea = Imgproc.contourArea(contour)
            
            // Calculate arc length manually to avoid MatOfPoint2f compatibility issues
            var arcLength = 0.0
            for (i in 0 until points.size - 1) {
                val p1 = points[i]
                val p2 = points[i + 1]
                val dx = p2.x - p1.x
                val dy = p2.y - p1.y
                arcLength += kotlin.math.sqrt(dx * dx + dy * dy)
            }
            
            val pointCount = points.size
            
            // Enhanced intensity calculation considering both area and arc length for open contours
            val baseIntensity = when {
                // For very small areas, use arc length to determine intensity (open contours)
                contourArea < 5 && arcLength > 15 -> 20f   // Open contours with good length
                contourArea < 5 && arcLength > 25 -> 30f   // Longer open contours
                contourArea < 10 -> 15f                    // Small closed contours
                contourArea < 30 -> 25f                    // Medium contours
                contourArea < 80 -> 35f                    // Large contours
                else -> 50f                                // Very large contours
            }
            
            // Bonus intensity for contours with many points (detailed shapes/open contours)
            val detailBonus = if (pointCount > 20) 5f else 0f
            val finalIntensity = baseIntensity + detailBonus
            
            Log.d("OverlayView", "Contour - area: $contourArea, arc: $arcLength, points: $pointCount, intensity: $finalIntensity")
            
            // For open contours (low area, high arc length), use line-based heat distribution
            if (contourArea < 10 && arcLength > 15) {
                // Draw heat along the contour path for open contours
                for (i in 0 until points.size - 1) {
                    val p1 = points[i]
                    val p2 = points[i + 1]
                    drawLineHeat(heatmap, width, height, p1, p2, finalIntensity)
                }
            } else {
                // Traditional area-based heat for closed contours
                for (point in points) {
                    val centerX = point.x.toInt()
                    val centerY = point.y.toInt()
                    
                    // Larger radius for more visible heatmap
                    for (dy in -4..4) {
                        for (dx in -4..4) {
                            val x = centerX + dx
                            val y = centerY + dy
                            
                            if (x >= 0 && x < width && y >= 0 && y < height) {
                                val index = (y * width) + x
                                if (index < heatmap.size) {
                                    // Gentler distance-based falloff for larger heat spread
                                    val distance = sqrt((dx * dx + dy * dy).toDouble()).toFloat()
                                    val intensity = finalIntensity * (1f / (1f + distance * 0.2f))
                                    
                                    heatmap[index] = min(maxHeatmapValue, heatmap[index] + intensity)
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Debug: Check final heatmap values
        val maxHeat = heatmap.maxOrNull() ?: 0f
        val nonZeroCount = heatmap.count { it > 0f }
        if (maxHeat > 0) {
            Log.d("OverlayView", "Heatmap updated - Max value: $maxHeat, Non-zero pixels: $nonZeroCount")
        }
        
        heatmapAge[faceKey] = currentTime
    }
    
    private fun createTestHeatData(faceKey: FaceRect, faceRect: org.opencv.core.Rect) {
        val width = faceRect.width
        val height = faceRect.height
        
        Log.d("OverlayView", "Creating test heat data for face ${width}x${height}")
        
        // Initialize heatmap array if it doesn't exist
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(width * height) { 0f }
            heatmapData[faceKey] = heatmap
            Log.d("OverlayView", "Created new heatmap array for test: ${width}x${height} = ${heatmap.size} pixels")
        }
        
        // Create some test heat patterns to verify heatmap rendering
        val centerX = width / 2
        val centerY = height / 2
        val radius = min(width, height) / 4
        
        // Create a circular heat pattern
        for (y in 0 until height) {
            for (x in 0 until width) {
                val distance = kotlin.math.sqrt(((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY)).toDouble()).toFloat()
                if (distance <= radius) {
                    val index = y * width + x
                    if (index < heatmap.size) {
                        val intensity = 30f * (1f - distance / radius) // Strong test heat
                        heatmap[index] = min(maxHeatmapValue, heatmap[index] + intensity)
                    }
                }
            }
        }
        
        val currentTime = System.currentTimeMillis()
        heatmapAge[faceKey] = currentTime
        
        val maxHeat = heatmap.maxOrNull() ?: 0f
        val nonZeroCount = heatmap.count { it > 0f }
        Log.d("OverlayView", "Test heat data created - Max value: $maxHeat, Non-zero pixels: $nonZeroCount")
    }
    
    private fun createForceHeatData(faceKey: FaceRect, faceRect: org.opencv.core.Rect) {
        val width = faceRect.width
        val height = faceRect.height
        
        Log.d("OverlayView", "FORCE creating heat data for face ${width}x${height}")
        
        // Always create new heatmap array
        val heatmap = FloatArray(width * height) { 0f }
        
        // Create multiple heat spots for guaranteed visibility
        val spots = listOf(
            Pair(width / 4, height / 4),     // Top-left
            Pair(3 * width / 4, height / 4), // Top-right  
            Pair(width / 2, height / 2),     // Center
            Pair(width / 4, 3 * height / 4), // Bottom-left
            Pair(3 * width / 4, 3 * height / 4) // Bottom-right
        )
        
        for ((spotX, spotY) in spots) {
            // Create heat around each spot
            for (y in max(0, spotY - 10) until min(height, spotY + 10)) {
                for (x in max(0, spotX - 10) until min(width, spotX + 10)) {
                    val index = y * width + x
                    if (index < heatmap.size) {
                        heatmap[index] = 50f // Very strong heat
                    }
                }
            }
        }
        
        heatmapData[faceKey] = heatmap
        heatmapAge[faceKey] = System.currentTimeMillis()
        
        val maxHeat = heatmap.maxOrNull() ?: 0f
        val nonZeroCount = heatmap.count { it > 0f }
        Log.d("OverlayView", "FORCE heat data created - Max value: $maxHeat, Non-zero pixels: $nonZeroCount")
    }
    
    private fun createHeatFromDifference(faceKey: FaceRect, faceRect: org.opencv.core.Rect, diff: Mat) {
        val width = faceRect.width
        val height = faceRect.height
        
        Log.d("OverlayView", "Creating heat from difference image for face ${width}x${height}")
        
        // Initialize heatmap array
        var heatmap = heatmapData[faceKey]
        if (heatmap == null) {
            heatmap = FloatArray(width * height) { 0f }
            heatmapData[faceKey] = heatmap
        }
        
        // Sample pixels from the difference image to create heat
        for (y in 0 until height step 2) { // Sample every 2nd pixel for performance
            for (x in 0 until width step 2) {
                val pixelValue = diff.get(y, x)[0] // Get difference value
                if (pixelValue > 2.0) { // Very low threshold for any visible difference
                    val index = y * width + x
                    if (index < heatmap.size) {
                        val intensity = (pixelValue / 255.0 * 20.0).toFloat() // Scale to heat intensity
                        heatmap[index] = min(maxHeatmapValue, heatmap[index] + intensity)
                        
                        // Add heat to neighboring pixels for better visibility
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                val nx = x + dx
                                val ny = y + dy
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                    val neighborIndex = ny * width + nx
                                    if (neighborIndex < heatmap.size) {
                                        heatmap[neighborIndex] = min(maxHeatmapValue, heatmap[neighborIndex] + intensity * 0.5f)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        heatmapAge[faceKey] = System.currentTimeMillis()
        
        val maxHeat = heatmap.maxOrNull() ?: 0f
        val nonZeroCount = heatmap.count { it > 0f }
        if (maxHeat > 0) {
            Log.d("OverlayView", "Diff heat created - Max value: $maxHeat, Non-zero pixels: $nonZeroCount")
        }
    }
    
    private fun drawLineHeat(heatmap: FloatArray, width: Int, height: Int, p1: Point, p2: Point, intensity: Float) {
        // Draw heat along a line between two points for open contours
        val x1 = p1.x.toInt()
        val y1 = p1.y.toInt()
        val x2 = p2.x.toInt()
        val y2 = p2.y.toInt()
        
        // Use Bresenham's line algorithm to draw heat along the line
        val dx = kotlin.math.abs(x2 - x1)
        val dy = kotlin.math.abs(y2 - y1)
        val sx = if (x1 < x2) 1 else -1
        val sy = if (y1 < y2) 1 else -1
        var err = dx - dy
        
        var x = x1
        var y = y1
        
        while (true) {
            // Apply heat around the current line point
            for (dy2 in -2..2) {
                for (dx2 in -2..2) {
                    val nx = x + dx2
                    val ny = y + dy2
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        val index = (ny * width) + nx
                        if (index < heatmap.size) {
                            val distance = sqrt((dx2 * dx2 + dy2 * dy2).toDouble()).toFloat()
                            val lineIntensity = intensity * (1f / (1f + distance * 0.3f))
                            heatmap[index] = min(maxHeatmapValue, heatmap[index] + lineIntensity)
                        }
                    }
                }
            }
            
            if (x == x2 && y == y2) break
            
            val e2 = 2 * err
            if (e2 > -dy) {
                err -= dy
                x += sx
            }
            if (e2 < dx) {
                err += dx
                y += sy
            }
        }
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
                // Slow decay for better visibility
                val decayRate = 0.99f // Slow decay 
                for (i in entry.value.indices) {
                    entry.value[i] *= decayRate
                }
            }
        }
    }
    
    // Object tracking functions (replicated from Python version)
    private fun findMostConsistentObject(): FaceRect? {
        if (objectHistory.size < minConsistencyFrames) {
            return null
        }
        
        // Count appearances for each object with position tolerance
        val objectCounts = mutableMapOf<String, Pair<Int, FaceRect>>()
        
        for (frameObjects in objectHistory) {
            for (obj in frameObjects) {
                // Create a position-based key (rounded to reduce exact position dependency)
                val posKey = "${(obj.left / 10).toInt() * 10},${(obj.top / 10).toInt() * 10},${((obj.right - obj.left) / 10).toInt() * 10},${((obj.bottom - obj.top) / 10).toInt() * 10}"
                val currentCount = objectCounts[posKey]?.first ?: 0
                objectCounts[posKey] = Pair(currentCount + 1, obj)
            }
        }
        
        // Find object with highest count
        val mostConsistent = objectCounts.maxByOrNull { it.value.first }
        return if (mostConsistent != null && mostConsistent.value.first >= minConsistencyFrames) {
            mostConsistent.value.second
        } else null
    }
    
    private fun shouldResetConsistency(): Boolean {
        if (currentObjects.isEmpty()) {
            consistencyResetCounter++
        } else {
            consistencyResetCounter = 0
        }
        return consistencyResetCounter >= maxResetFrames
    }
    
    private fun calculateDistance(rect1: FaceRect, rect2: FaceRect): Float {
        val center1X = rect1.left + (rect1.right - rect1.left) / 2
        val center1Y = rect1.top + (rect1.bottom - rect1.top) / 2
        val center2X = rect2.left + (rect2.right - rect2.left) / 2
        val center2Y = rect2.top + (rect2.bottom - rect2.top) / 2
        
        return kotlin.math.sqrt(
            ((center1X - center2X) * (center1X - center2X) + 
             (center1Y - center2Y) * (center1Y - center2Y)).toDouble()
        ).toFloat()
    }
    
    private fun updateConsistentFacePosition(consistentFace: FaceRect?, mostConsistentObject: FaceRect?): FaceRect? {
        if (consistentFace == null || currentObjects.isEmpty() || mostConsistentObject == null) {
            return consistentFace
        }
        
        // Find the best match in current objects (closest to most consistent)
        var bestMatch: FaceRect? = null
        var bestDistance = Float.MAX_VALUE
        
        for (obj in currentObjects) {
            val distance = calculateDistance(mostConsistentObject, obj)
            if (distance < bestDistance) {
                bestDistance = distance
                bestMatch = obj
            }
        }
        
        if (bestMatch == null) return consistentFace
        
        // Calculate movement speed based on distance (Python version logic)
        val currentCenterX = consistentFace.left + (consistentFace.right - consistentFace.left) / 2
        val currentCenterY = consistentFace.top + (consistentFace.bottom - consistentFace.top) / 2
        val targetCenterX = bestMatch.left + (bestMatch.right - bestMatch.left) / 2
        val targetCenterY = bestMatch.top + (bestMatch.bottom - bestMatch.top) / 2
        
        val distance = kotlin.math.sqrt(
            ((currentCenterX - targetCenterX) * (currentCenterX - targetCenterX) + 
             (currentCenterY - targetCenterY) * (currentCenterY - targetCenterY)).toDouble()
        ).toFloat()
        
        // Distance-based position update speed (matching Python logic)
        val posAlpha = when {
            distance < 10f -> 0.02f // Very close - 98% new position (very fast)
            distance < 25f -> 0.1f  // Close - 90% new position (fast)
            distance < 50f -> 0.25f // Medium - 75% new position (medium)
            distance < 80f -> 0.5f  // Far - 50% new position (slow)
            distance < 120f -> 0.8f // Very far - 20% new position (very slow)
            else -> 0.95f           // Extremely far - 5% new position (minimal)
        }
        
        // Apply position and size updates
        val updatedLeft = posAlpha * consistentFace.left + (1f - posAlpha) * bestMatch.left
        val updatedTop = posAlpha * consistentFace.top + (1f - posAlpha) * bestMatch.top
        val updatedRight = posAlpha * consistentFace.right + (1f - posAlpha) * bestMatch.right
        val updatedBottom = posAlpha * consistentFace.bottom + (1f - posAlpha) * bestMatch.bottom
        
        return FaceRect(updatedLeft.toInt(), updatedTop.toInt(), updatedRight.toInt(), updatedBottom.toInt())
    }
    
    private fun drawHeatmap(canvas: Canvas, faceKey: FaceRect, heatmap: FloatArray) {
        val width = faceKey.right - faceKey.left
        val height = faceKey.bottom - faceKey.top
        
        if (width <= 0 || height <= 0) return
        
        val paint = Paint()
        paint.style = Paint.Style.FILL
        paint.isAntiAlias = true
        
        // Maximum resolution for best visibility
        for (y in 0 until height step 1) { // Single pixel step for maximum resolution
            for (x in 0 until width step 1) { // Single pixel step for maximum resolution
                val index = y * width + x
                if (index < heatmap.size) {
                    val intensity = heatmap[index] / maxHeatmapValue
                    if (intensity > 0.005f) { // Ultra-low threshold for maximum sensitivity
                        val color = getHeatmapColor(intensity)
                        paint.color = color
                        
                        val screenX = faceKey.left + x
                        val screenY = faceKey.top + y
                        
                        // Draw larger pixels for better visibility
                        val pixelSize = 3f // Larger pixel size
                        canvas.drawRect(
                            (screenX * uniformScaleFactor) + xOffset,
                            (screenY * uniformScaleFactor) + yOffset,
                            ((screenX + pixelSize) * uniformScaleFactor) + xOffset,
                            ((screenY + pixelSize) * uniformScaleFactor) + yOffset,
                            paint
                        )
                    }
                }
            }
        }
    }
    
    private fun getHeatmapColor(intensity: Float): Int {
        // Maximum visibility color gradient - very bright and opaque
        val alpha = (intensity * 255).toInt().coerceIn(150, 255) // Much more opaque, minimum 150 alpha
        
        return when {
            intensity < 0.1f -> {
                // Bright cyan for even low activity
                Color.argb(alpha, 0, 200, 255)
            }
            intensity < 0.3f -> {
                // Bright green for low-medium activity  
                Color.argb(alpha, 0, 255, 150)
            }
            intensity < 0.5f -> {
                // Bright yellow for medium activity
                Color.argb(alpha, 255, 255, 0)
            }
            intensity < 0.7f -> {
                // Bright orange for medium-high activity
                Color.argb(alpha, 255, 150, 0)
            }
            else -> {
                // Bright red for high activity
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