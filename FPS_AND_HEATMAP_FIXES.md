# FPS and Heatmap Visibility Fixes

## Issues Addressed

### 1. **MediaPipe Detection Interval**
**Changed**: From 0.3 seconds to 0.1 seconds
```kotlin
// BEFORE
private val detectionInterval = 300L // 0.3 seconds

// AFTER  
private val detectionInterval = 100L // 0.1 seconds
```

### 2. **Grayscale FPS Matching Live Video**
**Fixed**: Contrast detection now runs on every frame for full FPS
```kotlin
// BEFORE (Limited FPS)
if (shouldDetectWithMediaPipe) {
    detectAsync(mpImage, frameTime)
} else {
    onFrameForContrastDetection(rotatedBitmap) // Only between detections
}

// AFTER (Full FPS)
// Always run contrast detection on every frame for full FPS
onFrameForContrastDetection(rotatedBitmap)

// Only run MediaPipe face detection every 0.1 seconds  
if (shouldDetectWithMediaPipe) {
    detectAsync(mpImage, frameTime)
}
```

### 3. **Heatmap Visibility Improvements**

#### **Enhanced Resolution**
- **Step size**: Changed from 4 pixels to 2 pixels (4x better resolution)
- **Lower threshold**: From 0.1f to 0.05f intensity (more sensitive)
- **Anti-aliasing**: Added for smoother rendering

#### **Improved Color Scheme**
```kotlin
// Enhanced 5-stage color gradient:
intensity < 0.2f → Blue (more visible blue minimum)
intensity < 0.4f → Cyan  
intensity < 0.6f → Green
intensity < 0.8f → Yellow
intensity ≥ 0.8f → Red

// Increased opacity: 60-180 alpha (was 0-120)
```

#### **Better Intensity Scaling**
```kotlin
// BEFORE
Small contours: 2f intensity
Medium contours: 4f intensity  
Large contours: 7f intensity
Very large: 10f intensity

// AFTER (More Visible)
Small contours: 5f intensity (+150%)
Medium contours: 8f intensity (+100%)  
Large contours: 12f intensity (+71%)
Very large: 15f intensity (+50%)
```

#### **Slower Decay Rate**
```kotlin
// BEFORE
val decayRate = 0.98f // Fast fade

// AFTER  
val decayRate = 0.995f // Slower fade for better visibility
```

### 4. **Drawing Order Optimization**
```kotlin
// BEFORE (Heatmap hidden)
Draw heatmap first
Draw grayscale bitmap (opaque) → Hides heatmap

// AFTER (Heatmap visible)
Draw grayscale bitmap (alpha 200) → Semi-transparent
Draw heatmap on top → Fully visible
```

### 5. **Debug Visualization**
Added real-time heatmap intensity indicator:
```kotlin
// Shows "H:15" etc. when heatmap data exists
if (maxValue > 1f) {
    canvas.drawText("H:${maxValue.toInt()}", x, y, paint)
}
```

## Performance Optimizations

### **Frame Processing**
- **MediaPipe**: Every 0.1s (10 FPS) for face position only
- **Contrast detection**: Every frame (30+ FPS) for motion analysis
- **Heatmap rendering**: Real-time with optimized 2-pixel sampling

### **Memory Efficiency**
- **Grayscale bitmap**: Semi-transparent rendering (alpha 200)
- **Heatmap data**: Optimized storage with intelligent decay
- **Resource cleanup**: Proper Mat management maintained

## Expected Results

### **Visual Improvements**
- ✅ **Heatmap clearly visible** over grayscale face regions
- ✅ **Smooth color gradient** (blue → cyan → green → yellow → red)
- ✅ **Higher resolution** heatmap (2-pixel vs 4-pixel sampling)
- ✅ **Better opacity** for visibility (60-180 alpha range)

### **Performance Improvements**  
- ✅ **Full FPS contrast detection** (matches live video)
- ✅ **0.1s MediaPipe intervals** for face position updates
- ✅ **Real-time motion tracking** with enhanced sensitivity
- ✅ **Smooth heatmap animation** with slower decay

### **Debug Features**
- ✅ **Heatmap intensity indicator** (green "H:##" text)
- ✅ **Real-time verification** of heatmap data existence
- ✅ **Visual confirmation** of motion detection activity

## System Architecture

### **Processing Pipeline**
1. **Every Frame (30+ FPS)**:
   - Camera input → Grayscale conversion
   - Contrast enhancement (histogram eq + normalization + gamma + unsharp)
   - Frame difference calculation
   - Adaptive dynamic thresholding
   - Contour detection and filtering
   - Heatmap data update
   - Real-time visualization

2. **Every 0.1 seconds (10 FPS)**:
   - MediaPipe face detection
   - Face region boundary update
   - New face tracking initialization

### **Rendering Order**
1. Semi-transparent grayscale face bitmap (alpha 200)
2. Heatmap overlay with enhanced colors
3. Debug intensity indicator  
4. Face detection box (green outline)
5. Eye detection rectangles
6. Detection confidence text

The system now provides **full FPS contrast detection** with **highly visible heatmaps** while maintaining **efficient 0.1s MediaPipe face position updates**.