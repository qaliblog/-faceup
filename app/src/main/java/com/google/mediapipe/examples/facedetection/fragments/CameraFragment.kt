package com.google.mediapipe.examples.facedetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
// import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.facedetection.FaceDetectorHelper
// import com.google.mediapipe.examples.facedetection.MainViewModel
import com.google.mediapipe.examples.facedetection.R
import com.google.mediapipe.examples.facedetection.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraFragment : Fragment(), FaceDetectorHelper.DetectorListener {

    private val TAG = "FaceDetection"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var faceDetectorHelper: FaceDetectorHelper
    // private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var currentBitmap: Bitmap? = null

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService
    
    private var isFaceDetectorInitialized = false

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(),
                R.id.fragment_container
            )
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }

        if(isFaceDetectorInitialized && this::faceDetectorHelper.isInitialized && this::backgroundExecutor.isInitialized) {
            backgroundExecutor.execute {
                try {
                    if (faceDetectorHelper.isClosed()) {
                        faceDetectorHelper.setupFaceDetector()
                    }
                } catch (e: Exception) {
                    // Ignore setup errors in onResume
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()

        // Close the face detector and release resources
        if(isFaceDetectorInitialized && this::faceDetectorHelper.isInitialized && this::backgroundExecutor.isInitialized) {
            backgroundExecutor.execute { 
                try {
                    faceDetectorHelper.clearFaceDetector() 
                } catch (e: Exception) {
                    // Ignore cleanup errors
                }
            }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        isFaceDetectorInitialized = false
        super.onDestroyView()

        // Shut down our background executor.
        if(this::backgroundExecutor.isInitialized) {
            backgroundExecutor.shutdown()
            backgroundExecutor.awaitTermination(
                Long.MAX_VALUE,
                TimeUnit.NANOSECONDS
            )
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Create the FaceDetectionHelper that will handle the inference
        backgroundExecutor.execute {
            try {
                faceDetectorHelper =
                    FaceDetectorHelper(
                        context = requireContext(),
                        faceDetectorListener = this@CameraFragment,
                        runningMode = RunningMode.LIVE_STREAM
                    )
                
                isFaceDetectorInitialized = true
                
                // Only set up camera after faceDetectorHelper is initialized
                activity?.runOnUiThread {
                    if (_fragmentCameraBinding != null && isAdded && isFaceDetectorInitialized) {
                        // Wait for the views to be properly laid out
                        fragmentCameraBinding.viewFinder.post {
                            // Set up the camera and its use cases
                            setUpCamera()
                        }
                    }
                }
            } catch (e: Exception) {
                isFaceDetectorInitialized = false
                activity?.runOnUiThread {
                    if (isAdded) {
                        Toast.makeText(requireContext(), "Failed to initialize face detector: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        // Ensure faceDetectorHelper is initialized before proceeding
        if (!isFaceDetectorInitialized || !this::faceDetectorHelper.isInitialized) {
            return
        }

        // CameraProvider
        val cameraProvider =
            cameraProvider
                ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build()

        // Preview. Set the aspect ratio to 16:9
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work, set aspect ratio to 16:9
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(
                        backgroundExecutor,
                        faceDetectorHelper::detectLivestreamFrame
                    )
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }


   private fun getRotationCompensation(): Int {
        val rotation = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            requireActivity().display?.rotation ?: 0
        } else {
            (requireActivity().getSystemService(android.content.Context.WINDOW_SERVICE) as? android.view.WindowManager)?.defaultDisplay?.rotation ?: 0
        }
        return when (rotation) {
            android.view.Surface.ROTATION_90 -> 90
            android.view.Surface.ROTATION_180 -> 180
            android.view.Surface.ROTATION_270 -> 270
            else -> 0
        }
    }
    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(-rotationDegrees.toFloat())
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }

    // Update UI after faces have been detected. Extracts original image height/width
    // to scale and place bounding boxes properly through OverlayView
    override fun onResults(resultBundle: FaceDetectorHelper.ResultBundle) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                try {
                    // Pass necessary information to OverlayView for drawing on the canvas
                    val detectionResult = resultBundle.results[0]
                    var bitmap = resultBundle.bitmap
                    val rotation = getRotationCompensation()
                    if(rotation != 0 && bitmap != null){
                        bitmap = rotateBitmap(bitmap, rotation)
                    }
                    if (bitmap != null) {
                        fragmentCameraBinding.overlay.setResults(
                            detectionResult,
                            bitmap.height,
                            bitmap.width,
                            bitmap
                        )
                    }

                    // Force a redraw
                    fragmentCameraBinding.overlay.invalidate()
                } catch (e: Exception) {
                    // Fragment might be destroyed, ignore
                }
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onFrameForContrastDetection(bitmap: Bitmap?) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                try {
                    fragmentCameraBinding.overlay.processContrastDetection(bitmap)
                } catch (e: Exception) {
                    // Fragment might be destroyed, ignore
                }
            }
        }
    }
}