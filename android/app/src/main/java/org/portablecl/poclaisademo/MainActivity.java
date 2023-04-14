package org.portablecl.poclaisademo;


import static android.hardware.camera2.CameraMetadata.LENS_FACING_BACK;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraCharacteristics.Key;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.portablecl.poclaisademo.databinding.ActivityMainBinding;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {

    // These native functions are defined in src/main/cpp/vectorAddExample.cpp
    // todo: move these headers to their own file
    public native int initCL();

    public native int vectorAddCL(int N, float[] A, float[] B, float[] C);

    public native int destroyCL();

    public native void setPoCLEnv(String key, String value);

    /**
     * NOTE: many of these variables are declared in this scope
     * because different callback methods need access to them and
     * therefore can not be passed as an argument
     */

    /**
     * the camera device that will be used to capture images
     */
    private CameraDevice chosenCamera;

    /**
     * used to create capture requests for the camera
     */
    private CaptureRequest.Builder requestBuilder;

    /**
     * used to control capturing images from the camera
     */
    private CameraCaptureSession captureSession;

    /**
     * the dimensions that the camera will be capturing images in
     */
    private Size captureSize;

    /**
     * a semaphore to prevent the camera from being closed at the wrong time
     */
    private final Semaphore cameraLock = new Semaphore(1);

    /**
     * the image reader is used to get images for processing
     */
    private ImageReader imageReader;

    /**
     * the view where to show previews on
     */
    private TextureView previewView;

    /**
     * a thread to run things in background and not block the UI thread
     */
    private HandlerThread backgroundThread;

    /**
     * a Handler that is used to schedule work on the backgroundThread
     */
    private Handler backgroundThreadHandler;

    /**
     * set how verbose the program should be
     */
    private static int verbose;

    /**
     * if true, each function will print when they are being called
     */
    private static final boolean DEBUGEXECUTION = true;

    /**
     * a list permissions to request
     */
    private static final String[] required_permissions = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    // Used to load the 'poclaisademo' library on application startup.
    static {
        System.loadLibrary("poclaisademo");
//        todo: check that this load is actually needed
        System.loadLibrary("poclremoteexample");
    }

    TextView ocl_text;
    TextView pocl_text;

    private ActivityMainBinding binding;

    /**
     * see https://developer.android.com/guide/components/activities/activity-lifecycle
     * what the purpose of this function is.
     *
     * @param savedInstanceState If the activity is being re-initialized after
     *                           previously being shut down then this Bundle contains the data it most
     *                           recently supplied in {@link #onSaveInstanceState}.  <b><i>Note: Otherwise it is null.</i></b>
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // for activities you should call their parent methods as well
        super.onCreate(savedInstanceState);

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onCreate method");
        }

        // todo: make these configurable
        verbose = 1;
        captureSize = new Size(640, 480);

        // get camera permission
        // (if not yet gotten, the screen will show a popup to grant permissions)
        // also need to be declared in the androidmanifest
        requestPermissions(required_permissions, PackageManager.PERMISSION_GRANTED);

        // get display things
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        // set the root content on top
        setContentView(binding.getRoot());

        ocl_text = binding.clOutput;
        previewView = binding.cameraFeed;

        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };
        td.start();

        // todo: have this be a parameter set by the user
        String server_address = "192.168.50.112";

        String cache_dir = getCacheDir().getAbsolutePath();

        //configure environment variables pocl needs to run
//        setPoCLEnv("POCL_DEBUG", "all");
//        setPoCLEnv("POCL_DEVICES", "remote");
//        setPoCLEnv("POCL_REMOTE0_PARAMETERS", server_address);
//        setPoCLEnv("POCL_CACHE_DIR", cache_dir);

    }

    /**
     * this is the last function is called before things are actually running.
     * <p>
     * see https://developer.android.com/guide/components/activities/activity-lifecycle
     * what the purpose of this function is.
     */
    @Override
    protected void onResume() {
        super.onResume();

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onResume method");
        }

        startBackgroundThread();

        // when the app starts, the previewView might not be available yet,
        // in that case, do nothing and wait for on resume to be called again
        // once the preview is available
        if (previewView.isAvailable()) {
            Log.println(Log.INFO, "MA flow", "preview available, setting up camera");
            setupCamera();
        } else {
            Log.println(Log.INFO, "MA flow", "preview not available, not setting up camera");
        }
    }

    /**
     * this is the first function is called when stopping an activity
     * <p>
     * see https://developer.android.com/guide/components/activities/activity-lifecycle
     * what the purpose of this function is.
     */
    @Override
    protected void onPause() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onPause method");
        }

        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    /**
     * call the opencl native code
     */
    void doVectorAdd() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started doVectorAdd method");
        }

        printLog(ocl_text, "\ncalling opencl init functions... ");
        initCL();

        // Create 2 vectors A & B
        // And yes, this array size is embarrassingly huge for demo!
        float[] A = {1, 2, 3, 4, 5, 6, 7};
        float[] B = {8, 9, 0, 6, 7, 8, 9};
        float[] C = new float[A.length];

        printLog(ocl_text, "\n A: ");
        for (int i = 0; i < A.length; i++)
            printLog(ocl_text, A[i] + "    ");

        printLog(ocl_text, "\n B: ");
        for (int i = 0; i < B.length; i++)
            printLog(ocl_text, B[i] + "    ");

        printLog(ocl_text, "\n\ncalling opencl vector-addition kernel... ");
        vectorAddCL(C.length, A, B, C);

        printLog(ocl_text, "\n C: ");
        for (int i = 0; i < C.length; i++)
            printLog(ocl_text, C[i] + "    ");

        boolean correct = true;
        for (int i = 0; i < C.length; i++) {
            if (C[i] != (A[i] + B[i])) {
                correct = false;
                break;
            }
        }

        if (correct)
            printLog(ocl_text, "\n\nresult: passed\n");
        else
            printLog(ocl_text, "\n\nresult: failed\n");

        printLog(ocl_text, "\ndestroy opencl resources... ");
        destroyCL();
    }


    void printLog(TextView tv, final String str) {
        // UI updates should happen only in UI thread
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tv.append(str);
            }
        });
    }

    /**
     * helper function to print all available options of a given camera
     *
     * @param manager
     * @param camera_id
     * @throws CameraAccessException
     */
    void printCameraCharacteristics(CameraManager manager, String camera_id) throws CameraAccessException {


        printLog(ocl_text, " camera " + camera_id + ": \n");

        CameraCharacteristics currentCharacteristics = manager.getCameraCharacteristics(camera_id);
        String chars = "    camera " + camera_id + ": \n";

        List<Key<?>> available_keys = currentCharacteristics.getKeys();
        for (Key key : available_keys) {
            chars = chars + key.toString() + " " + currentCharacteristics.get(key).toString() + " \n";
        }
        Log.println(Log.INFO, "Camera", chars + "\n ");

    }

    /**
     * select and setup the camera.
     * you should first get the camera permission before calling this.
     */
    private void setupCamera() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started setupCamera method");
        }

        // android want you to check permissions before calling openCamera, might as well request permissions again if not granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.println(Log.WARN, "MainActivity.java:setupCamera", "no camera permission, requesting permission and exiting function");
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PackageManager.PERMISSION_GRANTED);
            return;
        }

        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        try {

            if (verbose >= 3) {
                for (String cameraId : cameraManager.getCameraIdList()) {
                    printCameraCharacteristics(cameraManager, cameraId);
                }
            }

            // find a suitable camera
            for (String cameraId : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

                // skip if the camera is not back facing
                if (!(characteristics.get(CameraCharacteristics.LENS_FACING) == LENS_FACING_BACK)) {
                    continue;
                }

                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                Size[] sizes = map.getOutputSizes(ImageFormat.YUV_420_888);

                if (verbose >= 3) {
                    for (Size size : sizes) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera", "available size of camera " + cameraId + ": " + size.toString());
                    }
                }

                if (!Arrays.asList(sizes).contains(captureSize)) {
                    if (verbose >= 2) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera", "camera " + cameraId + "does not have requested size of " + captureSize.toString());
                    }
                    continue;
                }

                if (!cameraLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                    Log.println(Log.WARN, "MainActivity.java:setupcamera", "could not acquire camera lock ");
                } else {
                    // finally get the camera
                    cameraManager.openCamera(cameraId, cameraStateCallback, backgroundThreadHandler);
                }
                break;
            }
        } catch (CameraAccessException e) {
            Log.println(Log.WARN, "MainActivity.java:setupcamera", "could not access camera");
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * this callback object is attached to the camera and handles different scenarios
     */
    private final CameraDevice.StateCallback cameraStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "CameraDevice onOpened callback called");
            }
            cameraLock.release();
            chosenCamera = camera;
            createPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "CameraDevice onDisconnected callback called");
            }
            cameraLock.release();
            camera.close();
            chosenCamera = null;
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            cameraLock.release();
            camera.close();
            chosenCamera = null;
            Log.println(Log.ERROR, "MainActivity.java:cameraStateCallback", "camera device got into error state");
        }
    };

    /**
     * link the camera to the texture can start capturing camera images
     */
    private void createPreview() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started createPreview method");
        }

        try {

            SurfaceTexture previewTexture = previewView.getSurfaceTexture();
            assert previewTexture != null;

            // todo: maybe set size to scale with camera size
            previewTexture.setDefaultBufferSize(captureSize.getWidth(), captureSize.getHeight());

            Surface previewSurface = new Surface(previewTexture);

            assert chosenCamera != null;
            requestBuilder = chosenCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            requestBuilder.addTarget(previewSurface);

            // todo: add the imagereader to output surfaces
            // todo: use the new proper method for creating a capture session
            // https://stackoverflow.com/questions/67077568/how-to-correctly-use-the-new-createcapturesession-in-camera2-in-android
            chosenCamera.createCaptureSession(Collections.singletonList(previewSurface), previewStateCallback, null);

        } catch (CameraAccessException e) {
            Log.println(Log.ERROR, "MainActivity.java:createPreview", "could not access camera");
            e.printStackTrace();
        }
    }

    /**
     * a callback used to make a capture request once the camera is configured
     */
    private final CameraCaptureSession.StateCallback previewStateCallback = new CameraCaptureSession.StateCallback() {
        @Override
        public void onConfigured(@NonNull CameraCaptureSession session) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "CameraCaptureSession onConfigured callback called");
            }
            assert chosenCamera != null;
            assert requestBuilder != null;

            captureSession = session;

            try {
                requestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                CaptureRequest captureRequest = requestBuilder.build();

                // todo: possibly add listener to request
                captureSession.setRepeatingRequest(captureRequest, null, backgroundThreadHandler);

            } catch (CameraAccessException e) {
                Log.println(Log.ERROR, "MainActivity.java:previewStateCallback", "failed to set repeated request");
                e.printStackTrace();
            }

        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
            Log.println(Log.ERROR, "MainActivity.java:previewStateCallback", "failed to configure camera");
        }
    };


    /**
     * close the chosen camera and everything related to it
     */
    private void closeCamera() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started closeCamera method");
        }
        try {
            cameraLock.acquire();
            if (null != captureSession) {
                captureSession.close();
                captureSession = null;
            }
            if (null != chosenCamera) {
                chosenCamera.close();
                chosenCamera = null;
            }
            if (null != imageReader) {
                imageReader.close();
                imageReader = null;
            }
        } catch (Exception e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            cameraLock.release();
        }
    }


    /**
     * function to start the backgroundThread + handler
     */
    private void startBackgroundThread() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startBackgroundThread method");
        }
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundThreadHandler = new Handler(backgroundThread.getLooper());
    }

    /**
     * function to safely stop backgroundThread + handler
     */
    private void stopBackgroundThread() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started stopBackgroundThread method");
        }
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundThreadHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}