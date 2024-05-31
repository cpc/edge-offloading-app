package org.portablecl.poclaisademo;


import static android.hardware.camera2.CameraMetadata.LENS_FACING_BACK;
import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLELOGGINGKEY;
import static org.portablecl.poclaisademo.BundleKeys.LOGKEYS;
import static org.portablecl.poclaisademo.BundleKeys.TOTALLOGS;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.LOCAL_DEVICE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.REMOTE_DEVICE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.allCompressionOptions;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.getCompressionString;
import static org.portablecl.poclaisademo.JNIutils.setNativeEnv;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
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
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.Size;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.portablecl.poclaisademo.databinding.ActivityMainBinding;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
    private static boolean orientationsSwapped;

    /**
     * a semaphore to prevent the camera from being closed at the wrong time
     */
    private final Semaphore cameraLock = new Semaphore(1);

    /**
     * a semaphore used to sync the image process loop with available images.
     * zero starting permits, so a lock can only be acquired when the
     * camerareader releases a permit.
     */
    private final Semaphore imageAvailableLock = new Semaphore(0);

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

    private String IPAddress;
    private boolean disableRemote;

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

    private HandlerThread cameraLogThread;

    private Handler cameraLogHandler;


    /**
     * boolean to enable logging that gets set during creation
     */
    private boolean enableLogging;

    /**
     * Quality parameter of camera's JPEG compression
     */
    private int jpegQuality;

    private int configFlags;

    /**
     * file descriptor needed to open logging file
     */
    private final ParcelFileDescriptor[] parcelFileDescriptors =
            new ParcelFileDescriptor[TOTALLOGS];

    /**
     * used to write to logging file
     */
    private final FileOutputStream[] logStreams = new FileOutputStream[TOTALLOGS];

    /**
     * uri to the logging file
     */
    private final Uri[] uris = new Uri[TOTALLOGS];

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
     * object to keep track of ping times.
     */
    private static PingMonitor pingMonitor;

    private static PoclImageProcessor poclImageProcessor;

    /**
     * used to schedule a thread to periodically update stats
     */
    private ScheduledExecutorService statUpdateScheduler;

    /**
     * needed to stop the stat update thread
     */
    private ScheduledFuture statUpdateFuture;

    /**
     * needed to stop the stat logging thread
     */
    private ScheduledFuture statLoggerFuture;

    private CameraLogger cameraLogger;

    private ConfigStore configStore;

    // Used to load the 'poclaisademo' library on application startup.
    static {
        System.loadLibrary("poclaisademo");
    }

    static TextView ocl_text;
    TextView pocl_text;

    private ActivityMainBinding binding;

    private Switch modeSwitch;

    private DropEditText qualityText;

    private StatLogger statLogger;
    private DiscoverySelect DSSelect;

    /**
     * a dropdown item that can be used to set the compression to be used
     */
    private Spinner compressionSpinner;

    /**
     * the list of items in the compresssionSpinner
     */
    private ArrayAdapter<String> compressionEntries;
    final boolean[] discoveryReconnectCheck = {true};

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

        // used to retrieve settings set by the StartupActivity
        configStore = new ConfigStore(this);
        configFlags = configStore.getConfigFlags();
        jpegQuality = configStore.getJpegQuality();
        IPAddress = configStore.getIpAddressText();
        // get bundle with variables set during startup activity
        Bundle bundle = getIntent().getExtras();

        disableRemote = bundle.getBoolean(DISABLEREMOTEKEY);

        try {
            enableLogging = bundle.getBoolean(ENABLELOGGINGKEY, false);
        } catch (Exception e) {
            if (VERBOSITY >= 2) {
                Log.println(Log.INFO, "mainactivity:logging", "could not read enablelogging");
            }
            enableLogging = false;
        }

        for (int i = 0; i < TOTALLOGS; i++) {
            if (enableLogging) {

                try {
                    uris[i] = Uri.parse(bundle.getString(LOGKEYS[i], null));
                } catch (Exception e) {
                    if (VERBOSITY >= 2) {
                        Log.println(Log.INFO, "mainactivity:logging", "could not parse uri");
                    }
                    uris[i] = null;
                }
            }
            parcelFileDescriptors[i] = null;
            logStreams[i] = null;

        }

        // todo: make these configurable
        captureSize = new Size(640, 480);
        imageBufferSize = 35;

        if ((JPEG_IMAGE & configFlags) > 0) {
            captureFormat = ImageFormat.JPEG;
        } else {
            captureFormat = ImageFormat.YUV_420_888;
        }

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

        modeSwitch = binding.modeSwitch;
        modeSwitch.setOnClickListener(modeListener);

        Switch segmentationSwitch = binding.segmentSwitch;
        segmentationSwitch.setOnClickListener(segmentListener);

        compressionSpinner = binding.compressionSpinner;
        ArrayList<String> spinnerentries = populateSpinnerEntries(allCompressionOptions,
                configFlags);
        compressionEntries = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, spinnerentries);
        compressionSpinner.setAdapter(compressionEntries);
        // default to no compression
        compressionSpinner.setSelection(compressionEntries.getPosition("no compression"));
        compressionSpinner.setOnItemSelectedListener(compressionSpinnerListener);

        // setup overlay
        overlayVisualizer = new OverlayVisualizer();
        overlayView = binding.overlayView;
        // TODO: see if there is a better way to do this
        overlayView.setZOrderOnTop(true);

        Context context = getApplicationContext();
        energyMonitor = new EnergyMonitor(context);
        trafficMonitor = new TrafficMonitor();
        statLogger = new StatLogger(null,
                trafficMonitor, energyMonitor, pingMonitor);

        counter = new FPSCounter();
        statUpdateScheduler = Executors.newScheduledThreadPool(2);

        setNativeEnv("POCL_DISCOVERY", "0");
        Spinner discoverySpinner = binding.discoverySpinner;

        DSSelect = new DiscoverySelect(this, discoverySpinner, discoverySpinnerListener);

        if (disableRemote) {
            // disable this switch when remote is disabled
            modeSwitch.setClickable(false);
            setNativeEnv("POCL_DEVICES", "pthread");
        } else if (IPAddress == null || IPAddress.isEmpty()) {
            setNativeEnv("POCL_DEVICES", "pthread proxy");
            modeSwitch.setClickable(false);
        } else {
            modeSwitch.setClickable(true);
            setNativeEnv("POCL_DEVICES", "pthread proxy remote remote");
            setNativeEnv("POCL_REMOTE0_PARAMETERS", IPAddress + "/0");
            setNativeEnv("POCL_REMOTE1_PARAMETERS", IPAddress + "/1");
        }

        String nativeLibraryPath = this.getApplicationInfo().nativeLibraryDir;
        setNativeEnv("LD_LIBRARY_PATH", nativeLibraryPath);

        // disable pocl logs if verbosity is 0
        if (VERBOSITY >= 1) {
            setNativeEnv("POCL_DEBUG", "basic,pthread,proxy,error,debug,warning");
        }

        // TODO: Use this to copy the .onnx files there on startup (or try getFilesDir())
        String cache_dir = getCacheDir().getAbsolutePath();
        setNativeEnv("POCL_CACHE_DIR", cache_dir);


        // stop screen from turning off
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        boolean enableQualityAlgorithm = configStore.getQualityAlgorithmOption();
        int targetFPS = configStore.getTargetFPS();
        int pipelineLanes = configStore.getPipelineLanes();
        poclImageProcessor = new PoclImageProcessor(this, captureSize, null, captureFormat,
                imageAvailableLock, configFlags, counter, LOCAL_DEVICE,
                segmentationSwitch.isChecked(), uris[0],
                statLogger, enableQualityAlgorithm, targetFPS, pipelineLanes);

        // code to handle the quality input
        qualityText = binding.compressionEditText;
        qualityText.setOnEditorActionListener(qualityTextListener);
        qualityText.setOnFocusChangeListener(qualityFocusListener);
        qualityText.setText(Integer.toString(jpegQuality));
        poclImageProcessor.setQuality(jpegQuality);

        // when jpeg_image is enabled, the camera only outputs jpegs,
        // so offloading with compression is the only option
        if ((JPEG_IMAGE & configFlags) > 0) {
            modeSwitch.performClick();
            modeSwitch.setClickable(false);
            qualityText.setClickable(false);
            qualityText.setFocusable(false);
        }

        // uncomment this if you always want to start with remote on
//        modeSwitch.performClick();

        // if the quality algorithm is on, the user has no say
        // so disable all the buttons
        if (enableQualityAlgorithm) {
            compressionSpinner.setClickable(false);
            compressionSpinner.setFocusable(false);
            compressionSpinner.setAllowClickWhenDisabled(false);
            modeSwitch.setClickable(false);
            qualityText.setClickable(false);
        }

        // TODO: remove this example
        // this is an example run
        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };
        td.start();

    }

    /**
     * a function that parses the configflags and returns a list of compression options
     *
     * @param allOptions  list of all available compression options
     * @param configFlags the config flags
     * @return list of compression options
     */
    private ArrayList<String> populateSpinnerEntries(HashMap<String, Integer> allOptions,
                                                     int configFlags) {
        ArrayList<String> returnList = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : allOptions.entrySet()) {
            if ((entry.getValue() & configFlags) > 0) {
                returnList.add(entry.getKey());
            }
        }

        return returnList;
    }

    /**
     * callback function that handles users selecting a remote server out of list of discovered
     * servers. It also updates the activity's pingmonitor to the current servers pingmonitor.
     * Also responsible for handling the behaviour of modeswitch in case of no remote servers.
     */
    private final AdapterView.OnItemSelectedListener discoverySpinnerListener =
            new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position,
                                           long id) {
                    String selectedServer = DSSelect.spinnerList.get(position).getAddress();
                    if (DSSelect.serviceMap.containsKey(IPAddress) && discoveryReconnectCheck[0]) {
                        DSSelect.serviceMap.get(IPAddress).reconnect = true;

                        for (DiscoverySelect.spinnerObject value : DSSelect.spinnerList) {
                            if (value.address.equals(IPAddress)) {
                                pingMonitor = value.pingMonitor;
                            }
                        }
                        discoveryReconnectCheck[0] = false;
                    }
                    if (!selectedServer.equals(DiscoverySelect.DEFAULT_SPINNER_VAL)) {
                        modeSwitch.setClickable(true);
                        Log.d("DISC", "Spinner position selected: " + position + " : server " +
                                "selected " +
                                ": " + selectedServer);
                        DiscoverySelect.serviceInfo temp =
                                DSSelect.serviceMap.get(selectedServer);
                        Log.d("DISC", "Reconnect status: " + temp.reconnect);
                        poclImageProcessor.stop();
                        Discovery.addDevice(selectedServer + "/0", (temp.reconnect ? 1 : 0));
                        Discovery.addDevice(selectedServer + "/1", (temp.reconnect ? 1 : 0));
                        temp.reconnect = true;
                        poclImageProcessor.start(temp.name);
                        pingMonitor = DSSelect.spinnerList.get(position).pingMonitor;
                    } else if (modeSwitch.isChecked()) {
                        modeSwitch.performClick();
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> parent) {
                    modeSwitch.setClickable(IPAddress != null);
                }
            };

    /**
     * callback function that handles users changing the compression options
     */
    private final AdapterView.OnItemSelectedListener compressionSpinnerListener =
            new AdapterView.OnItemSelectedListener() {

                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position,
                                           long id) {

                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started " +
                                "compressionSpinnerListener " +
                                "onItemSelected");
                    }

                    // if we are doing things locally only, and the user tries to change things,
                    // set the position back to no compression
                    int noCompIndex = compressionEntries.getPosition("no compression");
                    if (!modeSwitch.isChecked() && position != noCompIndex) {

                        compressionSpinner.setSelection(noCompIndex);
                        return;
                    }

                    String entry = (String) parent.getItemAtPosition(position);
                    int compressionType = allCompressionOptions.get(entry);
                    poclImageProcessor.setCompressionType(compressionType);

                }

                @Override
                public void onNothingSelected(AdapterView<?> parent) {
                    Log.println(Log.ERROR, "compressionSpinnerListener", "onnothingselected " +
                            "callback " +
                            "called");
                }
            };

    /**
     * A callback that handles the quality edittext on screen when it loses focus.
     * This callback checks the input and sets it within the bounds of 0 - 100.
     * It also passes this input to the poclimageprocessor
     */
    private final View.OnFocusChangeListener qualityFocusListener =
            new View.OnFocusChangeListener() {
                @Override
                public void onFocusChange(View v, boolean hasFocus) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started qualityTextListener " +
                                "callback");
                    }

                    if (!hasFocus) {
                        TextView textView = (DropEditText) v;
                        int qualityInput;
                        try {
                            qualityInput = Integer.parseInt(textView.getText().toString());
                        } catch (Exception e) {
                            if (VERBOSITY >= 3) {
                                Log.println(Log.INFO, "MainActivity.java", "could not parse " +
                                        "quality, " +
                                        "defaulting to 80");
                            }
                            qualityInput = 80;
                            textView.setText(Integer.toString(qualityInput));
                        }

                        if (qualityInput < 1) {
                            qualityInput = 1;
                            textView.setText(Integer.toString(qualityInput));
                        } else if (qualityInput > 100) {
                            qualityInput = 100;
                            textView.setText(Integer.toString(qualityInput));
                        }

                        poclImageProcessor.setQuality(qualityInput);
                    }
                }
            };

    /**
     * A callback that loses focus when the done button is pressed on a TextView.
     */
    private final TextView.OnEditorActionListener qualityTextListener =
            new TextView.OnEditorActionListener() {
                @Override
                public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started qualityTextListener " +
                                "callback");
                    }

                    if (EditorInfo.IME_ACTION_DONE == actionId) {
                        v.clearFocus();
                    }
                    return false;
                }
            };

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
            if (VERBOSITY >= 2) {
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

    private void resetMonitors() {
        counter.reset();
        energyMonitor.reset();
        trafficMonitor.reset();
        poclImageProcessor.resetLastIou();
        if (pingMonitor != null) {
            pingMonitor.reset();
        }

    }

    /**
     * A listener that hands interactions with the mode switch.
     * This switch sets the device variable and resets monitors
     */
    private final View.OnClickListener modeListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (((Switch) v).isChecked()) {
                Toast.makeText(MainActivity.this, "Switching to remote device, please wait",
                        Toast.LENGTH_SHORT).show();
                poclImageProcessor.setInferencingDevice(REMOTE_DEVICE);
            } else {
                Toast.makeText(MainActivity.this, "Switching to local device, please wait",
                        Toast.LENGTH_SHORT).show();
                poclImageProcessor.setInferencingDevice(LOCAL_DEVICE);
            }

            resetMonitors();
        }
    };

    /**
     * A listener that hands interactions with the segmentation switch.
     * This switch sets the segmentation variable and resets monitors
     */
    private final View.OnClickListener segmentListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (((Switch) v).isChecked()) {
                Toast.makeText(MainActivity.this, "enabling segmentation, please wait",
                        Toast.LENGTH_SHORT).show();
                poclImageProcessor.setDoSegment(true);
            } else {
                Toast.makeText(MainActivity.this, "disabling segmentation, please wait",
                        Toast.LENGTH_SHORT).show();
                poclImageProcessor.setDoSegment(false);
            }

            resetMonitors();
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

        resetMonitors();

        // schedule the metrics to update every second
        statUpdateFuture = statUpdateScheduler.scheduleAtFixedRate(statUpdater, 1, 1,
                TimeUnit.SECONDS);

        // when the app starts, the previewView might not be available yet,
        // in that case, do nothing and wait for on resume to be called again
        // once the preview is available
        if (previewView.isAvailable()) {
            Log.println(Log.INFO, "MA flow", "preview available, setting up camera");

            if (enableLogging) {
                boolean streamRes = openFileOutputStream(1);

                if (streamRes) {
                    statLogger.setStream(logStreams[1]);
                } else {
                    Log.println(Log.WARN, "Logging", "could not open file, disabling logging");
                    enableLogging = false;
                }

            }

            // todo: settle on desired period
            statLoggerFuture =
                    statUpdateScheduler.scheduleAtFixedRate(statLogger,
                            1000, 500, TimeUnit.MILLISECONDS);

            setupCamera();
            poclImageProcessor.start();


        } else {
            Log.println(Log.INFO, "MA flow", "preview not available, not setting up camera");
        }

    }

    private boolean openFileOutputStream(int i) {
        assert (i < TOTALLOGS);
        try {
            parcelFileDescriptors[i] = getContentResolver().openFileDescriptor(uris[i], "wa");
            logStreams[i] = new FileOutputStream(parcelFileDescriptors[i].getFileDescriptor());

        } catch (Exception e) {
            Log.println(Log.WARN, "openFileOutputStreams", "could not open log file " + i
                    + ": " + uris[i].toString() + " :" + e);
            logStreams[i] = null;
            return false;
        }
        return true;
    }

    private void closeFileOutputStreams() {
        for (int i = 0; i < TOTALLOGS; i++) {
            if (null != logStreams[i]) {
                try {
                    logStreams[i].close();
                } catch (IOException e) {
                    Log.println(Log.WARN, "closeFileOutputStreams", "could not close " +
                            "fileoutputstream " + i);
                } finally {
                    logStreams[i] = null;
                }
            }

            if (null != parcelFileDescriptors[i]) {
                try {
                    parcelFileDescriptors[i].close();
                } catch (IOException e) {
                    Log.println(Log.WARN, "closeFileOutputStreams", "could not close " +
                            "parcelfiledescriptor " + i);
                } finally {
                    parcelFileDescriptors[i] = null;
                }
            }
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
                if (pingMonitor != null) {
                    pingMonitor.tick();
                }
                String formatString = "FPS: %3.1f (%4.0fms) AVG: %3.1f (%4.0fms)\n" +
                        "pow: %02.2f (%02.2f) W | EPF: %02.2f (%02.2f) J\n" +
                        new String(Character.toChars(0x1F50B)) + "time left:%3dm:%2ds |avg " +
                        "latency:%4.0f ms\n" +
                        "bandwidth: ∇ %s | ∆ %s\n" +
                        "ping: %5.1fms AVG: %5.1fms | IoU: %6.4f\n";

                float fps = counter.getEMAFPSTimespan();
                // fallback to the emafps for lower values since it captures it better
                if (fps < 3) {
                    fps = counter.getEMAFPS();
                }
                float avgfps = counter.getAverageFPS();
                float eps = -energyMonitor.getEMAEPS();
                float avgeps = -energyMonitor.getAverageEPS();
                float remainingTime = energyMonitor.estimateSecondsRemaining();
                int remainingMinutes = (int) (remainingTime / 60);
                int remainingSeconds = (int) (remainingTime % 60);
                float fpssecs = (0 != fps) ? 1000 / fps : 0;
                float avgfpssecs = (0 != avgfps) ? 1000 / avgfps : 0;
                float epf = (0 != fps) ? eps / fps : 0;
                float avgepf = (0 != avgfps) ? avgeps / avgfps : 0;
                float iou = poclImageProcessor.getLastIou();
                float emaLatency = counter.getEmaLatency() / 1000;

                // pingMonitor can be null because the pingreader is started when the mode switch
                // is pressed
                float ping = 0.0f;
                float ping_avg = 0.0f;
                if (pingMonitor != null && !pingMonitor.isReaderNull()) {
                    ping = pingMonitor.getPing();
                    ping_avg = pingMonitor.getAveragePing();
                }

                // get ping from fill buffer
//                Stats stats = getStats();
//                float ping = stats.pingMs;
//                float ping_avg = stats.pingMsAvg;

                String statString = String.format(Locale.US, formatString,
                        fps, fpssecs,
                        avgfps, avgfpssecs,
                        eps, avgeps,
                        epf, avgepf,
                        remainingMinutes, remainingSeconds, emaLatency,
                        trafficMonitor.getRXBandwidthString(),
                        trafficMonitor.getTXBandwidthString(),
                        (modeSwitch.isChecked() && pingMonitor != null) ?
                                pingMonitor.getPing() : 0,
                        (modeSwitch.isChecked() && pingMonitor != null) ?
                                pingMonitor.getAveragePing() : 0,
                        iou
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

    public void drawOverlay(int doSegment, int[] detectionResults, byte[] segmentationResults
            , Size captureSize, boolean orientationsSwapped) {
        runOnUiThread(() -> overlayVisualizer.drawOverlay(doSegment, detectionResults,
                segmentationResults, captureSize, orientationsSwapped, overlayView));
    }

    /**
     * A function that changes the state of all buttons on the screen depending on the given
     * config
     */
    public void setButtonsFromJNI(CodecConfig config) {

        runOnUiThread(() -> {

            // show if we are using remote or not
            if (modeSwitch.isChecked() != (REMOTE_DEVICE == config.deviceIndex)) {
                modeSwitch.performClick();
            }

            // set the quality text that corresponds to a algorithm config
            qualityText.setText(Integer.toString(config.configIndex));

            // set the spinner to the right position if it isn't already the same
            int position =
                    compressionEntries.getPosition(getCompressionString(config.compressionType));
            if (compressionSpinner.getSelectedItemPosition() != position) {
                Log.println(Log.ERROR, "test", "position being changed by algo");
                compressionSpinner.setSelection(position);
            }
        });

    }

    /**
     * Statistics from the image processing loop that we might be interested in displaying in the UI
     */
    static class Stats {
        public final float pingMs;
        public final float pingMsAvg;

        public Stats(float pingMs, float pingMsAvg) {
            this.pingMs = pingMs;
            this.pingMsAvg = pingMsAvg;
        }
    }

    /**
     * enable or disable the mode switch. when disabling it, also set the switch to not checked
     *
     * @param value true to allow users to use the mode switch otherwise disable it
     */
    public void enableRemote(boolean value) {

        runOnUiThread(() -> {
            modeSwitch.setClickable(value);

            // if we disable it and the switch is checked,
            // turn off remote
            if (!value && modeSwitch.isChecked()) {
                modeSwitch.performClick();
            }

        });
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

        // used to stop the stat update scheduler.
        statUpdateFuture.cancel(true);
        if (null != statLoggerFuture) {
            statLoggerFuture.cancel(true);
            statLoggerFuture = null;
        }

        // imageprocessthread depends on camera and background threads, so close this first
        poclImageProcessor.stop();
        closeCamera();
        stopBackgroundThreads();

        closeFileOutputStreams();

        super.onPause();
    }

    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started onDestroy method");
        }

        DSSelect.stopDiscovery();

        // restart the app if the main activity is closed
        Context applicationContext = getApplicationContext();
        Intent intent =
                applicationContext.getPackageManager().getLaunchIntentForPackage(
                        applicationContext.getPackageName());
        Intent restartIntent = Intent.makeRestartActivityTask(intent.getComponent());
        restartIntent.putExtra(DISABLEREMOTEKEY, disableRemote);
        restartIntent.putExtra(ENABLELOGGINGKEY, enableLogging);
        applicationContext.startActivity(restartIntent);
        Runtime.getRuntime().exit(0);

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

            if (VERBOSITY >= 3) {
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

                if (VERBOSITY >= 3) {
                    for (Size size : sizes) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera", "available size of" +
                                " camera " + cameraId + ": " + size.toString());
                    }
                }

                if (!Arrays.asList(sizes).contains(captureSize)) {
                    if (VERBOSITY >= 2) {
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

                    if (VERBOSITY >= 3) {
                        Log.println(Log.INFO, "MainActivity.java:setupCamera", "minimum frame " +
                                "duration: "
                                + map.getOutputMinFrameDuration(captureFormat, captureSize)
                        );
                    }

                    // orientation is only known after setupcameraoutput, so set it now
                    poclImageProcessor.setOrientation(orientationsSwapped);
                    poclImageProcessor.setImageReader(imageReader);

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
        imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundThreadHandler);
    }

    private final ImageReader.OnImageAvailableListener imageAvailableListener =
            new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    if (DEBUGEXECUTION && VERBOSITY >= 3) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "image available");
                    }
                    imageAvailableLock.release();
                    if (imageAvailableLock.availablePermits() > 33) {
                        Image image = imageReader.acquireLatestImage();
                        imageAvailableLock.drainPermits();
                        image.close();
                        if (VERBOSITY >= 2) {
                            Log.println(Log.WARN, "imageavailablelistener", "imagereader buffer " +
                                    "got really full");
                        }
                    }

                }
            };

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

        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, "previewfeed",
                    "optimal camera preview size is: " + previewSize.toString());
        }

        // change the previewView size to fit our chosen aspect ratio from the chosen capture size
        int currentOrientation = getResources().getConfiguration().orientation;
        if (currentOrientation == Configuration.ORIENTATION_LANDSCAPE) {
            if (VERBOSITY >= 2) {
                Log.println(Log.INFO, "previewfeed", "set aspect ratio normal ");
            }
            previewView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
        } else {

            if (VERBOSITY >= 2) {
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

            if (VERBOSITY >= 2) {
                Log.println(Log.INFO, "MainActivity.java:configureTransform", "in rotated surface" +
                        " with scale: " + transformScale);
            }

            previewView.setTransform(transformMatrix);

        } else if (rotation == Surface.ROTATION_180) {
            transformMatrix.postRotate(180, centerX, centerY);

            if (VERBOSITY >= 2) {
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

            if (VERBOSITY >= 2) {
                Log.println(Log.ERROR, "previewfeed", "previewsizes: " + previewSize.toString());
                Log.println(Log.ERROR, "previewfeed",
                        "previewviewsizes: " + previewView.getWidth() + "x" + previewView.getHeight());
            }


            requestBuilder = chosenCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            requestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            if (captureFormat == ImageFormat.JPEG) {
                requestBuilder.set(CaptureRequest.JPEG_QUALITY, (byte) jpegQuality);
                if (VERBOSITY >= 1) {
                    Log.println(Log.INFO, "previewfeed",
                            "Setting camera JPEG quality to " + jpegQuality);
                }
            }
            // uncomment this to set the target fps (only usable in auto-exposure mode, not sure if
            // it works at all)
//            requestBuilder.set(CONTROL_AE_TARGET_FPS_RANGE, new Range<Integer>(15, 15));
            // The following set the camera parameters manually:
//            requestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest
//            .CONTROL_AE_MODE_OFF);
//            requestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME,   1000000L); // 1 ms
//            requestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, 640); // value of android
//            .sensor.maxAnalogSensitivity
//            requestBuilder.set(CaptureRequest.SENSOR_FRAME_DURATION, 16665880L); // 60 FPS
//            (doesn't work)
//            requestBuilder.set(CaptureRequest.SENSOR_FRAME_DURATION, 33333333L); // 30 FPS
            requestBuilder.addTarget(previewSurface);

            Surface imageReaderSurface = imageReader.getSurface();
            requestBuilder.addTarget(imageReaderSurface);


            // todo: use the new proper method for creating a capture session
            // https://stackoverflow.com/questions/67077568/how-to-correctly-use-the-new-createcapturesession-in-camera2-in-android
            chosenCamera.createCaptureSession(Arrays.asList(previewSurface,
                            imageReaderSurface),
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

                    if (chosenCamera == null || requestBuilder == null) {
                        return;
                    }

                    captureSession = session;

                    try {

                        CaptureRequest captureRequest = requestBuilder.build();

                        cameraLogger = null;
                        if (enableLogging) {
                            // only set callback if the filestream is successfully opened
                            if (null != uris[2] && openFileOutputStream(2)) {
                                cameraLogger = new CameraLogger(logStreams[2]);
                            }
                        }

                        // todo: possibly add listener to request
                        captureSession.setRepeatingRequest(captureRequest, cameraLogger,
                                cameraLogHandler);

                    } catch (CameraAccessException e) {
                        Log.println(Log.ERROR, "MainActivity.java:previewStateCallback", "failed " +
                                "to set repeated request");
                        e.printStackTrace();
                    } catch (IllegalStateException e) {
                        // this exception can occur if a new preview is being made before the
                        // first one is done
                        if (VERBOSITY >= 3) {
                            Log.println(Log.ERROR, "MainActivity.java:previewStateCallback",
                                    "session no longer available");
                        }
                    }
                    cameraLock.release();

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

        cameraLogThread = new HandlerThread("CameraLoggingThread");
        cameraLogThread.start();
        cameraLogHandler = new Handler(cameraLogThread.getLooper());

    }

    /**
     * function to safely stop backgroundThread + handler
     */
    private void stopBackgroundThreads() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started stopBackgroundThread method");
        }
        backgroundThread.quitSafely();
        cameraLogThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundThreadHandler = null;

            cameraLogThread.join();
            cameraLogThread = null;
            backgroundThreadHandler = null;

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
