# MediaPipe + OpenCV Contrast Detection Implementation

## Overview
This implementation combines MediaPipe face detection with OpenCV contrast difference detection to create a dynamic heatmap visualization system.

## Key Features Implemented

### 1. MediaPipe Face Detection with 0.3s Interval
- **File**: `FaceDetectorHelper.kt`
- **Implementation**: Added `lastDetectionTime` and `detectionInterval` variables
- **Logic**: MediaPipe detection runs every 300ms (0.3 seconds)
- **Between intervals**: Frames are passed to OpenCV contrast detection

### 2. OpenCV Contrast Difference Detection
- **File**: `OverlayView.kt`
- **Method**: `performContrastDetection()`
- **Process**:
  1. Convert current frame to grayscale
  2. Calculate absolute difference with previous frame
  3. Apply dynamic thresholding
  4. Find contours in difference image
  5. Update heatmap data with contour locations

### 3. Dynamic Contrast Adjustment
- **Method**: `calculateDynamicThreshold()`
- **Logic**: 
  - Low activity (mean < 10): threshold = 15.0
  - Medium activity (mean < 30): threshold = 25.0  
  - High activity (mean >= 30): threshold = 40.0
- **Adaptive**: Threshold adjusts based on frame difference intensity

### 4. Heatmap Visualization
- **Method**: `drawHeatmap()` and `getHeatmapColor()`
- **Features**:
  - Accumulates heat values at contour locations
  - Color gradient: Blue (cold) → Cyan → Yellow → Red (hot)
  - Semi-transparent overlay with alpha blending
  - Performance optimized with 4-pixel sampling

### 5. Color Reduction/Fading
- **Method**: `decayHeatmap()`
- **Features**:
  - Automatic decay with 0.98f rate per frame
  - Time-based cleanup (5 second expiration)
  - Gradual fade effect as activity decreases

## Technical Implementation Details

### Face Region Tracking
- Stores detected face regions in `lastFaceRegions`
- Contrast detection only operates within face bounds
- Efficient region-of-interest processing

### Memory Management
- Proper OpenCV Mat cleanup with `.release()`
- Background thread processing to avoid UI blocking
- Cached bitmap management for performance

### Threading
- MediaPipe detection: Background executor
- Contrast detection: Coroutine scope with IO dispatcher  
- UI updates: Main thread with `withContext(Dispatchers.Main)`

### Performance Optimizations
- 4-pixel step sampling for heatmap rendering
- Efficient heatmap data structure (FloatArray)
- Minimal memory allocations in hot paths
- Proper Mat recycling

## Integration Points

### 1. FaceDetectorHelper Interface
```kotlin
interface DetectorListener {
    fun onError(error: String, errorCode: Int = OTHER_ERROR)
    fun onResults(resultBundle: ResultBundle)
    fun onFrameForContrastDetection(bitmap: Bitmap) // NEW
}
```

### 2. CameraFragment Updates
- Implements new `onFrameForContrastDetection()` method
- Calls `OverlayView.processContrastDetection()`

### 3. OverlayView Enhancements
- New `processContrastDetection()` public method
- Enhanced `draw()` method includes heatmap rendering
- Updated `setResults()` stores face regions for contrast detection

## Visual Output
- **Face detection boxes**: Green outlines (every 0.3s)
- **Eye detection**: Green rectangles (existing feature)
- **Heatmap overlay**: Color-coded activity visualization
- **Dynamic intensity**: Adapts to motion amount and speed
- **Fading effect**: Heat gradually dissipates over time

## Dependencies
- OpenCV 4.10.0 (already included)
- MediaPipe Tasks Vision 0.10.14 (existing)
- Kotlin Coroutines (existing)

## Files Modified
1. `FaceDetectorHelper.kt` - Added timing control and new interface method
2. `OverlayView.kt` - Added contrast detection, heatmap, and rendering
3. `CameraFragment.kt` - Added contrast detection handler

The implementation provides a real-time visualization of facial activity using contrast difference detection with dynamic thresholding and a color-coded heatmap that fades over time.