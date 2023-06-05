package org.portablecl.poclaisademo;


import static android.hardware.camera2.CameraMetadata.LENS_FACING_BACK;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.destroyPoclImageProcessor;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.initPoclImageProcessor;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclProcessYUVImage;
import static org.portablecl.poclaisademo.JNIutils.setNativeEnv;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraCharacteristics.Key;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.portablecl.poclaisademo.databinding.ActivityMainBinding;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;


public class MainActivity extends AppCompatActivity {

    // These native functions are defined in src/main/cpp/vectorAddExample.cpp
    // todo: move these headers to their own file
    public native int initCL(AssetManager am);

    public native int vectorAddCL(int N, float[] A, float[] B, float[] C);

    public native int destroyCL();

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
     * used to create a capture request that pocl uses
     */
    private CaptureRequest.Builder imageReaderRequestBuilder;

    /**
     * used to request the camera to take a new image for pocl
     */
    private CaptureRequest imageReaderCaptureRequest;

    /**
     * used to control capturing images from the camera
     */
    private CameraCaptureSession captureSession;

    /**
     * the dimensions that the camera will be capturing images in
     */
    private static Size captureSize;

    /**
     * the image format to save captures in
     */
    private int captureFormat;

    /**
     * used to indicate if the camera rotation is different from the display
     */
    private static boolean orientationsSwapped = false;

    /**
     * a semaphore to prevent the camera from being closed at the wrong time
     */
    private final Semaphore cameraLock = new Semaphore(1);

    /**
     * the image reader is used to get images for processing
     */
    private ImageReader imageReader;

    /**
     * the number of images to buffer for the device
     */
    private int imageBufferSize;

    /**
     * the view where to show previews on
     */
    private AutoFitTextureView previewView;

    /**
     * the size of the preview, this is NOT the size of the previewView,
     * but rather the maximum largest camera capture size that has the right aspect ratio
     */
    private Size previewSize;

    /**
     * the bitmap to save the result of processing in
     */
    private static Bitmap resultBitmap;

    private static OverlayVisualizer overlayVisualizer;

    private static SurfaceView overlayView;

    /**
     * a thread to run things in background and not block the UI thread
     */
    private HandlerThread backgroundThread;

    /**
     * a Handler that is used to schedule work on the backgroundThread
     */
    private Handler backgroundThreadHandler;

    /**
     * a thread to run pocl on
     */
    private Thread imageProcessThread;

    /**
     * which device to use for inferencing
     */
    private int inferencing_device;
    private final int LOCAL_DEVICE = 0;

    private final int PASSTHRU_DEVICE = 1;
    private final int REMOTE_DEVICE = 2;



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

    /**
     * a counter to keep track of FPS metrics for our image processing.
     */
    private static FPSCounter counter;

    /**
     * an object used to keep track of the battery.
     */
    private static EnergyMonitor energyMonitor;

    /**
     * helper object for holding network traffic stats
     */
    private static TrafficMonitor trafficMonitor;

    /**
     * used to schedule a thread to periodically update stats
     */
    private ScheduledExecutorService statUpdateScheduler;

    /**
     * needed to stop the stat update thread
     */
    private ScheduledFuture statUpdateFuture;

    // Used to load the 'poclaisademo' library on application startup.
    static {
        System.loadLibrary("poclaisademo");
    }

    static TextView ocl_text;
    TextView pocl_text;

    private ActivityMainBinding binding;

    /**
     * see https://developer.android.com/guide/components/activities/activity-lifecycle
     * what the purpose of this function is.
     *
     * @param savedInstanceState If the activity is being re-initialized after
     *                           previously being shut down then this Bundle contains the data it
     *                           most
     *                           recently supplied in {@link #onSaveInstanceState}.  <b><i>Note:
     *                           Otherwise it is null.</i></b>
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // for activities you should call their parent methods as well
        super.onCreate(savedInstanceState);

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onCreate method");
        }

        // get bundle with variables set during startup activity
        Bundle bundle = getIntent().getExtras();

        // todo: make these configurable
        verbose = 1;
        captureSize = new Size(640, 480);
        imageBufferSize = 2;

        // this should be an image format we can work with on the native side.
        captureFormat = ImageFormat.YUV_420_888;

        // todo: check whether these args need to be updated.
        resultBitmap = Bitmap.createBitmap(captureSize.getWidth(), captureSize.getHeight(),
                Bitmap.Config.RGB_565);

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
        previewView.setSurfaceTextureListener(surfaceTextureListener);

        Switch modeSwitch = binding.modeSwitch;
        modeSwitch.setOnClickListener(modeListener);

        // setup overlay
        overlayVisualizer = new OverlayVisualizer();
        overlayView = binding.overlayView;
        // TODO: see if there is a better way to do this
        overlayView.setZOrderOnTop(true);

        Context context = getApplicationContext();
        energyMonitor = new EnergyMonitor(context);
        trafficMonitor = new TrafficMonitor();

        counter = new FPSCounter();
        statUpdateScheduler = Executors.newScheduledThreadPool(1);

        // TODO: remove this example
        // this is an example run
        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };
        td.start();


        String cache_dir = getCacheDir().getAbsolutePath();
        // used to configure pocl
        setNativeEnv("POCL_DEBUG", "basic,proxy,remote,error");
        setNativeEnv("POCL_DEVICES", "basic remote proxy");
        setNativeEnv("POCL_REMOTE0_PARAMETERS", bundle.getString("IP"));
        inferencing_device = LOCAL_DEVICE;
        setNativeEnv("POCL_CACHE_DIR", cache_dir);
    }

    /**
     * a listener to do things when the previewview changes
     */
    private final TextureView.SurfaceTextureListener surfaceTextureListener
            = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width,
                                              int height) {

        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width,
                                                int height) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "surfaceTextureListener " +
                        "onSurfaceTextureSizeChanged callback called");
            }
            if (verbose >= 2) {
                Log.println(Log.INFO, "previewfeed",
                        "callback with these params:" + width + "x" + height);
            }
            configureTransform(width, height);
            // it is possible that the preview is created before the resize takes effect,
            // making the feed warped, therefore, create preview again.
            try {
                cameraLock.acquire();
                createPreview();
                cameraLock.release();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {

        }
    };

    /**
     *  A listener that hands interactions with the mode switch.
     *  This switch sets the device variable and restarts the
     *  image process thread
     */
    private final View.OnClickListener modeListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (((Switch) v).isChecked()) {
                Toast.makeText(MainActivity.this, "Switching to remote device, please wait", Toast.LENGTH_SHORT).show();
                // TODO: uncomment this
                inferencing_device = REMOTE_DEVICE;
            } else {
                Toast.makeText(MainActivity.this, "Switching to local device, please wait", Toast.LENGTH_SHORT).show();
                inferencing_device = LOCAL_DEVICE;
            }
            //stopImageProcessThread();
            //startImageProcessThread();
        }
    };

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

        startBackgroundThreads();

        counter.Reset();
        energyMonitor.reset();
        trafficMonitor.reset();
        // schedule the metrics to update every second
        statUpdateFuture = statUpdateScheduler.scheduleAtFixedRate(statUpdater, 1, 1,
                TimeUnit.SECONDS);

        // when the app starts, the previewView might not be available yet,
        // in that case, do nothing and wait for on resume to be called again
        // once the preview is available
        if (previewView.isAvailable()) {
            Log.println(Log.INFO, "MA flow", "preview available, setting up camera");
            setupCamera();
            startImageProcessThread();

        } else {
            Log.println(Log.INFO, "MA flow", "preview not available, not setting up camera");
        }

    }

    private final Runnable statUpdater = new Runnable() {
        @Override
        public void run() {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "updating stats");
            }
            try {

                energyMonitor.tick();
                trafficMonitor.tick();
                String formatString = "FPS: %.2f  AVG FPS: %.2f \n" +
                        "EPS: %.3f W  AVG EPS: %.3f W \n" +
                        "charge: %d μAh\n" +
                        "voltage: %d mV \n" +
                        "current: %d mA\n" +
                        "bandwidth: ∇ %s | ∆ %s ";

                String statString = String.format(Locale.US, formatString,
                        counter.getFPS(),
                        counter.getAverageFPS(),
                        energyMonitor.getEPS(),
                        energyMonitor.getAverageEPS(),
                        energyMonitor.getCharge(),
                        energyMonitor.getVoltage(),
                        energyMonitor.getcurrent(),
                        trafficMonitor.getRXBandwidthString(),
                        trafficMonitor.getTXBandwidthString()
                );

                // needed since only the uithread is allowed to make changes to the textview
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        ocl_text.setText(statString);
                    }
                });

            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    };

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

        // used to stop the stat update scheduler.
        statUpdateFuture.cancel(false);

        closeCamera();
        stopBackgroundThreads();
        stopImageProcessThread();

        super.onPause();
    }

    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onDestroy method");
        }
        super.onDestroy();
    }

    /**
     * call the opencl native code
     */
    void doVectorAdd() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started doVectorAdd method");
        }

        printLog(ocl_text, "\ncalling opencl init functions... ");
        initCL(getAssets());

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
            chars = chars + key.toString() + " " + currentCharacteristics.get(key).toString() +
                    " \n";
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

        // android want you to check permissions before calling openCamera, might as well request
        // permissions again if not granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.println(Log.WARN, "MainActivity.java:setupCamera", "no camera permission, " +
                    "requesting permission and exiting function");
            requestPermissions(new String[]{Manifest.permission.CAMERA},
                    PackageManager.PERMISSION_GRANTED);
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
                CameraCharacteristics characteristics =
                        cameraManager.getCameraCharacteristics(cameraId);

                // skip if the camera is not back facing
                if (!(characteristics.get(CameraCharacteristics.LENS_FACING) == LENS_FACING_BACK)) {
                    continue;
                }

                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                Size[] sizes = map.getOutputSizes(captureFormat);

                if (verbose >= 3) {
                    for (Size size : sizes) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera", "available size of" +
                                " camera " + cameraId + ": " + size.toString());
                    }
                }

                if (!Arrays.asList(sizes).contains(captureSize)) {
                    if (verbose >= 2) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera",
                                "camera " + cameraId + " does not have requested size of " +
                                        captureSize.toString());
                    }
                    continue;
                }

                if (!cameraLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                    Log.println(Log.WARN, "MainActivity.java:setupcamera", "could not acquire " +
                            "camera lock ");
                } else {

                    setupImageReader();
                    setupCameraOutput(cameraId, previewView.getWidth(), previewView.getHeight());
                    // finally get the camera
                    cameraManager.openCamera(cameraId, cameraStateCallback,
                            backgroundThreadHandler);
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

    private void setupImageReader() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started setupImageReader method");
        }

        imageReader = ImageReader.newInstance(captureSize.getWidth(), captureSize.getHeight(),
                captureFormat, imageBufferSize);

    }

    /**
     * Method to call pocl. This method contains a while loop that exits
     * when an interrupt is sent to the thread running this method.
     * This method makes sure to start and destroy needed opencl objects.
     * This method also makes sure to queue an image capture for the next iteration.
     */
    private void imageProcessLoop() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started image process loop");
        }

        int MAX_DETECTIONS = 10;
        int MASK_W = 160;
        int MASK_H = 120;

        int detection_count = 1 + MAX_DETECTIONS * 6;
        int segmentation_count = MAX_DETECTIONS * MASK_W * MASK_H;

        Image image = null;
        try {

            initPoclImageProcessor(getAssets(), captureSize.getWidth(), captureSize.getHeight());

            // the main loop, will continue until an interrupt is sent
            while (!Thread.interrupted()) {

                if (DEBUGEXECUTION) {
                    Log.println(Log.INFO, "EXECUTIONFLOW", "started new image process" +
                            " iteration");
                }

                try {
                    // capture the next image.
                    // since the imagereader has a queue,
                    // we can request a new image to be captured,
                    // while working on another.
                    if (null != captureSession) {
                        captureSession.capture(imageReaderCaptureRequest, null, null);
                    }
                } catch (Exception e) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.WARN, "imageProcessLoop", "capture session is not available");
                    }
                }

                if (null != imageReader) {
                    image = imageReader.acquireLatestImage();
                }

                // if there wasn't an image available, sleep
                if (null == image) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "no image available, " +
                                "sleeping");
                    }
                    // if an interrupt is sent while sleeping,
                    // an execption is thrown. This is fine since
                    // we weren't doing anything anyway
                    Thread.sleep(17);
                    continue;
                }

                Image.Plane[] planes = image.getPlanes();

                ByteBuffer Y = planes[0].getBuffer();
                int YPixelStride = planes[0].getPixelStride();
                int YRowStride = planes[0].getRowStride();

                ByteBuffer U = planes[1].getBuffer();
                ByteBuffer V = planes[2].getBuffer();
                int UVPixelStride = planes[1].getPixelStride();
                int UVRowStride = planes[1].getRowStride();

                int VPixelStride = planes[2].getPixelStride();
                int VRowStride = planes[2].getRowStride();

                if (verbose >= 3) {

                    Log.println(Log.WARN, "imagereader", "plane count: " + planes.length);
                    Log.println(Log.WARN, "imagereader",
                            "Y pixel stride: " + YPixelStride);
                    Log.println(Log.WARN, "imagereader",
                            "Y row stride: " + YRowStride);
                    Log.println(Log.WARN, "imagereader",
                            "UV pixel stride: " + UVPixelStride);
                    Log.println(Log.WARN, "imagereader",
                            "UV row stride: " + UVRowStride);

                    Log.println(Log.WARN, "imagereader",
                            "V pixel stride: " + VPixelStride);
                    Log.println(Log.WARN, "imagereader",
                            "V row stride: " + VRowStride);
                }

                int[] detection_results = new int[detection_count];
                byte[] segmentation_results = new byte[segmentation_count * 4];
                int rotation = orientationsSwapped ? 90 : 0;
                poclProcessYUVImage(inferencing_device, rotation, Y, YRowStride,
                        YPixelStride, U, V, UVRowStride, UVPixelStride, detection_results, segmentation_results);

                runOnUiThread(() -> overlayVisualizer.drawOverlay(detection_results, segmentation_results,
                        captureSize, orientationsSwapped, overlayView));

                // used to calculate the (avg) FPS
                counter.TickFrame();

                // don't forget to close the image when done
                image.close();
            }

        } catch (InterruptedException e) {
            // if an image was open, close it.
            // can be null if the imagereader didn't have an image available
            if (image != null) {
                image.close();
            }
            Log.println(Log.INFO, "MainActivity.java:imageProcessLoop", "received " +
                    "interrupt, closing down");

        } catch (Exception e) {

            if (image != null) {
                image.close();
            }
            Log.println(Log.INFO, "MainActivity.java:imageProcessLoop", "error while " +
                    "processing image");
            e.printStackTrace();

        } finally {
            // always free pocl
            destroyPoclImageProcessor();
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "finishing image process loop");
            }
        }

    }

    /**
     * figure out the orientations of the camera and display and use that to setup the right
     * transformations
     * in order to make the preview not be warped.
     *
     * @param cameraId the camera id
     */
    private void setupCameraOutput(String cameraId, int width, int height) throws CameraAccessException {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started setupCameraOutput method");
        }

        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

        int displayOrientation = getWindowManager().getDefaultDisplay().getRotation();
        int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);

        switch (displayOrientation) {
            case Surface.ROTATION_0:
            case Surface.ROTATION_180: {
                if (sensorOrientation == 90 || sensorOrientation == 270) {
                    orientationsSwapped = true;
                }
                break;
            }
            case Surface.ROTATION_90:
            case Surface.ROTATION_270: {
                if (sensorOrientation == 0 || sensorOrientation == 180) {
                    orientationsSwapped = true;
                }
                break;
            }
            default: {
                Log.println(Log.ERROR, "MainActivity.java:setupCameraOutput", "unknown display " +
                        "orientation");
            }
        }

        Point displaySize = new Point();
        getWindowManager().getDefaultDisplay().getSize(displaySize);

        int rotatedPreviewWidth = width;
        int rotatedPreviewHeight = height;
        int maxPreviewWidth = displaySize.x;
        int maxPreviewHeight = displaySize.y;

        if (orientationsSwapped) {
            rotatedPreviewWidth = height;
            rotatedPreviewHeight = width;
            maxPreviewWidth = displaySize.y;
            maxPreviewHeight = displaySize.x;

        }

        // clip to the max width and height guaranteed by Camera2
        if (maxPreviewWidth > 1920) {
            maxPreviewWidth = 1920;
        }

        if (maxPreviewHeight > 1080) {
            maxPreviewHeight = 1080;
        }

        previewSize = chooseOptimalPreviewSize(cameraId, rotatedPreviewWidth,
                rotatedPreviewHeight, maxPreviewHeight, maxPreviewWidth);

        if (verbose >= 2) {
            Log.println(Log.INFO, "previewfeed",
                    "optimal camera preview size is: " + previewSize.toString());
        }

        // change the previewView size to fit our chosen aspect ratio from the chosen capture size
        int currentOrientation = getResources().getConfiguration().orientation;
        if (currentOrientation == Configuration.ORIENTATION_LANDSCAPE) {
            if (verbose >= 2) {
                Log.println(Log.INFO, "previewfeed", "set aspect ratio normal ");
            }
            previewView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
        } else {

            if (verbose >= 2) {
                Log.println(Log.INFO, "previewfeed", "set aspect ratio rotated ");
            }
            previewView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
        }
        configureTransform(width, height);

    }

    /**
     * set up a transformation for when the device is rotated.
     *
     * @param width  width of surface
     * @param height height of surface
     */
    private void configureTransform(int width, int height) {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started configureTransform method");
        }

        assert previewView != null;
        assert previewSize != null;

        int rotation = getWindowManager().getDefaultDisplay().getRotation();

        Matrix transformMatrix = new Matrix();
        RectF viewRect = new RectF(0, 0, width, height);
        RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());

        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();

        if (rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270) {

            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            // todo: check that src and dst should not be switched around
            transformMatrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.CENTER);

            float transformScale = Math.max(
                    (float) height / previewSize.getHeight(),
                    (float) width / previewSize.getWidth()
            );
            transformMatrix.postScale(transformScale, transformScale, centerX, centerY);
            transformMatrix.postRotate(90 * (rotation - 2), centerX, centerY);

            if (verbose >= 2) {
                Log.println(Log.INFO, "MainActivity.java:configureTransform", "in rotated surface" +
                        " with scale: " + transformScale);
            }

            previewView.setTransform(transformMatrix);

        } else if (rotation == Surface.ROTATION_180) {
            transformMatrix.postRotate(180, centerX, centerY);

            if (verbose >= 2) {
                Log.println(Log.INFO, "MainActivity.java:configureTransform", "in rotated 180 " +
                        "with scale: ");
            }

            previewView.setTransform(transformMatrix);
        }

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "setting transform now");
        }

        previewView.setTransform(transformMatrix);
    }

    /**
     * figure out the largest stream configuration that has the right aspect ratio,
     * but also isn't too large to start causing slowdowns
     *
     * @param cameraId
     * @param previewWidth
     * @param previewHeight
     * @param maxPreviewWidth
     * @param maxPreviewHeight
     * @return
     * @throws CameraAccessException
     */
    private Size chooseOptimalPreviewSize(String cameraId, int previewWidth, int previewHeight,
                                          int maxPreviewWidth, int maxPreviewHeight) throws CameraAccessException {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started chooseOptimalPreviewSize method");
        }

        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

        StreamConfigurationMap map = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        // its okey that the outputsizes we are getting here do not necessarily belong
        // to the chosen capture format, as long as they have the same capture size aspect ratio
        Size[] availableSizes = map.getOutputSizes(SurfaceTexture.class);

        List<Size> bigEnough = new ArrayList<>();
        List<Size> notBigEnough = new ArrayList<>();


        // sort the sizes into collections where items are greater than/equal and smaller
        for (Size size : availableSizes) {
            int sizeWidth = size.getWidth();
            int sizeHeight = size.getHeight();

            // continue if it is too big
            if (sizeWidth >= maxPreviewWidth || sizeHeight >= maxPreviewHeight) {
                continue;
            }

            // continue if not the right aspect ratio
            // order of math here is important so that we don't get rounding errors
            if (sizeHeight != (sizeWidth * captureSize.getHeight()) / captureSize.getWidth()) {
                continue;
            }

            if (sizeWidth >= previewWidth && sizeHeight >= previewHeight) {
                bigEnough.add(size);
            } else {
                notBigEnough.add(size);
            }

        }

        Comparator<Size> comparator = new CompareSizesByArea();
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, comparator);
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, comparator);
        } else {
            Log.println(Log.WARN, "MainActivity.java:chooseOptimalPreviewSize", "could not find " +
                    "optimal preview size");
            return availableSizes[0];
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size o1, Size o2) {

            return Integer.signum(o1.getWidth() * o1.getHeight() - o2.getWidth() * o2.getHeight());
        }
    }

    /**
     * this callback object is attached to the camera and handles different scenarios
     */
    private final CameraDevice.StateCallback cameraStateCallback =
            new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "CameraDevice onOpened callback " +
                                "called");
                    }

                    chosenCamera = camera;
                    createPreview();
                    cameraLock.release();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "CameraDevice onDisconnected " +
                                "callback called");
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
                    Log.println(Log.ERROR, "MainActivity.java:cameraStateCallback", "camera " +
                            "device got " +
                            "into error state");
                }
            };

    /**
     * link the camera to the texture can start capturing camera images
     *
     * @note: you need the camera lock before calling this.
     */
    private void createPreview() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started createPreview method");
        }

        if (chosenCamera == null) {
            return;
        }

        try {

            // this function needs to be called if the preview changes in size,
            // so it is possible there already is one open
            if (captureSession != null) {
                captureSession.close();
            }

            SurfaceTexture previewTexture = previewView.getSurfaceTexture();
            assert previewTexture != null;

            previewTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(previewTexture);

            if (verbose >= 2) {
                Log.println(Log.ERROR, "previewfeed", "previewsizes: " + previewSize.toString());
                Log.println(Log.ERROR, "previewfeed",
                        "previewviewsizes: " + previewView.getWidth() + "x" + previewView.getHeight());
            }


            requestBuilder = chosenCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            requestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            requestBuilder.addTarget(previewSurface);

            imageReaderRequestBuilder = chosenCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            imageReaderRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            imageReaderRequestBuilder.addTarget(imageReader.getSurface());

            // todo: use the new proper method for creating a capture session
            // https://stackoverflow.com/questions/67077568/how-to-correctly-use-the-new-createcapturesession-in-camera2-in-android
            chosenCamera.createCaptureSession(Arrays.asList(previewSurface,
                            imageReader.getSurface()),
                    previewStateCallback, null);
        } catch (CameraAccessException e) {
            Log.println(Log.ERROR, "MainActivity.java:createPreview", "could not access camera");
            e.printStackTrace();
        }
    }

    /**
     * a callback used to make a capture request once the camera is configured
     */
    private final CameraCaptureSession.StateCallback previewStateCallback =
            new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "CameraCaptureSession onConfigured" +
                                " " +
                                "callback called");
                    }

                    if (chosenCamera == null || requestBuilder == null ||
                            imageReaderRequestBuilder == null) {
                        return;
                    }

                    captureSession = session;

                    try {

                        CaptureRequest captureRequest = requestBuilder.build();
                        imageReaderCaptureRequest = imageReaderRequestBuilder.build();

                        // todo: possibly add listener to request
                        captureSession.setRepeatingRequest(captureRequest, null,
                                backgroundThreadHandler);

                    } catch (CameraAccessException e) {
                        Log.println(Log.ERROR, "MainActivity.java:previewStateCallback", "failed " +
                                "to set repeated request");
                        e.printStackTrace();
                    } catch (IllegalStateException e) {
                        // this exception can occur if a new preview is being made before the
                        // first one is done
                        if (verbose >= 3) {
                            Log.println(Log.ERROR, "MainActivity.java:previewStateCallback",
                                    "session no longer available");
                        }
                    }

                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                    Log.println(Log.ERROR, "MainActivity.java:previewStateCallback", "failed to " +
                            "configure camera");
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
    private void startBackgroundThreads() {
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
    private void stopBackgroundThreads() {
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

    /**
     * function to start the image process thread, and thereby the
     * main image process loop
     */
    private void startImageProcessThread() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startImageProcessThread method");
        }

        imageProcessThread = new Thread() {
            public void run() {
                imageProcessLoop();
            }
        };

        imageProcessThread.start();

    }

    /**
     * function to safely stop the image process thread and
     * ask it nicely to stop anything it is doing
     */
    private void stopImageProcessThread() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started stopImageProcessThread method");
        }
        if (null == imageProcessThread) {
            return;
        }

        // sending an interrupt will exit the while loop
        imageProcessThread.interrupt();
        try {
            // wait for the last iteration to be done
            imageProcessThread.join();
            imageProcessThread = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}