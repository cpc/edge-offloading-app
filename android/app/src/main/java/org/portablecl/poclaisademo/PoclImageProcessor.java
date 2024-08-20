package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.ENABLEFALLBACK;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.ENABLE_PROFILING;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.LOCAL_DEVICE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.LOCAL_ONLY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.SOFTWARE_HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.dequeue_spot;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.destroyPoclImageProcessorV2;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.getCodecConfig;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.initPoclImageProcessorV2;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclGetLastIouV2;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclSelectCodecAuto;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclSubmitYUVImage;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.receiveImage;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.waitImageAvailable;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.Semaphore;

public class PoclImageProcessor {

    public final static int MAX_FPS = 30;
    public final static int MAX_LANES = 64;
    /**
     * a semaphore used to sync the image process loop with available images.
     * zero starting permits, so a lock can only be acquired when the
     * camerareader releases a permit.
     */
    private final Semaphore imageAvailableLock;
    /**
     * the dimensions that the camera will be capturing images in
     */
    private final Size captureSize;
    /**
     * a counter to keep track of FPS metrics for our image processing.
     */
    private final FPSCounter counter;
    /**
     * required to get the asset manager and provide toasts
     */
    private final Context context;
    /**
     * the activity that is making use of the image processor,
     * used to update overlay stats
     */
    private final MainActivity activity;
    private final Uri uri;
    private final int configFlags;
    private final StatLogger statLogger;
    private final boolean enableQualityAlgorithm;
    private final boolean runtimeEval;
    private final boolean lockCodec;
    public int inferencingDevice;
    /**
     * the image reader is used to get images for processing
     */
    private ImageReader imageReader;
    /**
     * a thread to run pocl on
     */
    private Thread imageSubmitThread;
    private Thread receiverThread;
    private boolean orientationsSwapped;
    private boolean doSegment;
    private int compressionType;
    private int quality;
    private int imageFormat;
    private float lastIou = -4.0f;
    private int targetFPS;
    private int pipelineLanes;

    private final Uri vidUri;

    /**
     * constructor for pocl image processor
     *
     * @param context
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param configFlags
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     */
    public PoclImageProcessor(Context context, Size captureSize, ImageReader imageReader,
                              int imageFormat, Semaphore imageAvailableLock,
                              int configFlags,
                              FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment,
                              Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm,
                              boolean runtimeEval, boolean lockCodec,
                              int targetFPS, int pipelineLanes, Uri vidUri) {
        this(null, context, captureSize, imageReader, imageFormat, imageAvailableLock,
                configFlags,
                fpsCounter, inferencingDevice, doSegment, uri, statLogger,
                enableQualityAlgorithm, runtimeEval, lockCodec, targetFPS, pipelineLanes, vidUri);

    }

    /**
     * constructor for pocl image processor
     *
     * @param activity
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param configFlags
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     */
    public PoclImageProcessor(MainActivity activity, Size captureSize, ImageReader imageReader,
                              int imageFormat,
                              Semaphore imageAvailableLock, int configFlags,
                              FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment,
                              Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm,
                              boolean runtimeEval, boolean lockCodec,
                              int targetFPS, int pipelineLanes, Uri vidUri) {
        this(activity, activity, captureSize, imageReader, imageFormat, imageAvailableLock,
                configFlags, fpsCounter, inferencingDevice, doSegment, uri, statLogger,
                enableQualityAlgorithm, runtimeEval, lockCodec, targetFPS, pipelineLanes, vidUri);

    }

    /**
     * Constructor of the pocl image processor
     *
     * @param activity
     * @param context
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param configFlags
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     */
    private PoclImageProcessor(MainActivity activity, Context context, Size captureSize,
                               ImageReader imageReader, int imageFormat,
                               Semaphore imageAvailableLock, int configFlags,
                               FPSCounter fpsCounter,
                               int inferencingDevice, boolean doSegment,
                               Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm,
                               boolean runtimeEval, boolean lockCodec,
                               int targetFPS, int pipelineLanes, Uri vidUri) {

        this.activity = activity;
        this.context = context;
        this.captureSize = captureSize;
        this.imageReader = imageReader;
        this.imageAvailableLock = imageAvailableLock;
//        this.enableLogging = enableLogging;
        this.statLogger = statLogger;
        this.enableQualityAlgorithm = enableQualityAlgorithm;
        this.runtimeEval = runtimeEval;
        this.lockCodec = lockCodec;

        counter = fpsCounter;
        this.inferencingDevice = inferencingDevice;
        this.doSegment = doSegment;
        this.orientationsSwapped = false;
        this.uri = uri;

        this.configFlags = configFlags;
        // default value
        this.quality = 80;

        this.compressionType = NO_COMPRESSION;

        setImageFormat(imageFormat);

        this.imageSubmitThread = null;
        this.receiverThread = null;
        this.vidUri = vidUri;

        setTargetFPS(targetFPS);
        setPipelineLanes(pipelineLanes);

    }

    public static int sanitizeTargetFPS(int targetFPS) {
        if (targetFPS > MAX_FPS) {
            Log.println(Log.WARN, "PoclImageProcessor.java", "higher target than allowed, capping" +
                    " to MAX_FPS");
            targetFPS = MAX_FPS;
        } else if (targetFPS < 1) {
            Log.println(Log.WARN, "PoclImageProcessor.java", "lower target than allowed, capping " +
                    "to 1");
            targetFPS = 1;
        }
        return targetFPS;
    }

    public static int sanitizePipelineLanes(int lanes) {
        if (lanes > MAX_LANES) {
            Log.println(Log.WARN, "PoclImageProcessor.java", "higher target lanes than allowed, " +
                    "capping to MAX_LANES");
            lanes = MAX_LANES;
        } else if (lanes < 1) {
            Log.println(Log.WARN, "PoclImageProcessor.java", "lower target lanes than allowed, " +
                    "capping to 1");
            lanes = 1;
        }
        return lanes;
    }

    /**
     * indicate that the orientations are swapped
     *
     * @param orientation set to true if orientations are swapped
     */
    public void setOrientation(boolean orientation) {
        this.orientationsSwapped = orientation;
    }

    /**
     * Indicate which device to use.
     *
     * @param inferencingDevice
     */
    public void setInferencingDevice(int inferencingDevice) {
        this.inferencingDevice = inferencingDevice;
    }

    /**
     * Enable segmentation
     *
     * @param doSegment
     */
    public void setDoSegment(boolean doSegment) {
        this.doSegment = doSegment;
    }

    public void setCompressionType(int compressionType) {
        this.compressionType = compressionType;
    }

    public void setTargetFPS(int targetFPS) {

        this.targetFPS = sanitizeTargetFPS(targetFPS);
    }

    public void setPipelineLanes(int lanes) {

        this.pipelineLanes = sanitizePipelineLanes(lanes);
    }

    /**
     * Set the imageReader to use
     *
     * @param imageReader
     */
    public void setImageReader(ImageReader imageReader) {
        this.imageReader = imageReader;
    }

    public void setQuality(int quality) {
        this.quality = quality;
    }

    public void setImageFormat(int imageFormat) {

        // safety checks to make sure the image format is supported in this config
        assert ImageFormat.YUV_420_888 == imageFormat ||
                ImageFormat.JPEG == imageFormat;
        assert ImageFormat.YUV_420_888 != imageFormat || ((YUV_COMPRESSION & configFlags) > 0) ||
                ((JPEG_COMPRESSION & configFlags) > 0) ||
                ((NO_COMPRESSION & configFlags) > 0) ||
                ((HEVC_COMPRESSION & configFlags) > 0) ||
                ((SOFTWARE_HEVC_COMPRESSION & configFlags) > 0);
        assert ImageFormat.JPEG != imageFormat || ((JPEG_IMAGE & configFlags) > 0);

        this.imageFormat = imageFormat;
    }

    private boolean checkImageFormat(Image image) {
        int format = image.getFormat();
        // || (ImageFormat.JPEG == format);
        return (ImageFormat.YUV_420_888 == format);
    }

    public float getLastIou() {
        return this.lastIou;
    }

    public void resetLastIou() {
        this.lastIou = -4.0f;
    }

    /**
     * Method to call pocl. This method contains a while loop that exits
     * when an interrupt is sent to the thread running this method.
     * This method makes sure to start and destroy needed opencl objects.
     * This method also makes sure to queue an image capture for the next iteration.
     */
    private void imageProcessLoop(String serviceName) {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started image process loop");
        }

        int YPixelStride, YRowStride;
        int UVPixelStride, UVRowStride;
        int VPixelStride, VRowStride;
        int rotation, do_segment, do_algorithm, runtime_eval, lock_codec;
        long currentTime, doneTime, poclTime, imageAcquireTime;
        int size;
        float energy;
        int currentInferencingDevice; // inference device is needed both for
        // dequeue_spot and submit image, so copy value over to prevent the value changing
        int status = 0;
        // configflags that can change due to having to fallback
        int runtimeConfigFlags = this.configFlags;
        String runtimeServiceName = serviceName;

        Image image = null;

        ParcelFileDescriptor logParcelFd = null;
        ParcelFileDescriptor calibrateParcelFd = null;

        // used for frame limiting.
        // set a starting time that is definitely smaller
        // than the first time we check this value
        long lastFrameTime = System.nanoTime() - 10_000_000_000L;

        // work around to checking for interrupted multiple times
        // gets set when the while loop with the interrupt check exits
        boolean interrupted = false;

        // this outer while loop allows the loop to fallback to local execution if an error occurs
        while (!interrupted) {

            if (DEBUGEXECUTION) {
                Log.println(Log.WARN, "PoclImageProcessor.java:imageProcessLoop", "started a new " +
                        "process loop");
            }

            int logFd = -1;
            if ((ENABLE_PROFILING & runtimeConfigFlags) > 0) {
                try {
                    logParcelFd = context.getContentResolver().openFileDescriptor(uri,
                            "wa");
                } catch (FileNotFoundException e) {
                    Log.println(Log.WARN, "PoclImageProcessor.java:imageProcessLoop",
                            "could not open log filedescriptor");
                    return;
                }
                logFd = logParcelFd.getFd();
            }
            int calibrateFd = -1;
            if (null != vidUri) {
                try {
                    calibrateParcelFd = context.getContentResolver().openFileDescriptor(vidUri,
                            "r");
                    calibrateFd = calibrateParcelFd.getFd();
                } catch (Exception e) {
                    Log.println(Log.ERROR, "PoclImageProcessor.java:imageProcessLoop",
                            "could not open calibrate vid filedescriptor");
                    e.printStackTrace();
                    return;
                }
            }

            try {

                AssetManager assetManager = context.getAssets();
                do_algorithm = enableQualityAlgorithm ? 1 : 0;
                runtime_eval = runtimeEval ? 1 : 0;
                lock_codec = lockCodec ? 1 : 0;
                status = initPoclImageProcessorV2(runtimeConfigFlags, assetManager,
                        captureSize.getWidth(), captureSize.getHeight(), logFd,
                        this.pipelineLanes, do_algorithm, runtime_eval, lock_codec,
                        runtimeServiceName, calibrateFd);


                Log.println(Log.WARN, "temp ", " init return status: " + status);
                switch (status) {
                    case -100: {

                        if (ENABLEFALLBACK & !interrupted) {
                            runtimeConfigFlags = fallbackToLocal();
                            runtimeServiceName = null;
                            closeParcelFd(logParcelFd);
                            closeParcelFd(calibrateParcelFd);

                            // restart the whole process
                            continue;
                        }
                        // falldown to the next case when enablefallback is not enabled
                    }
                    case -33: {
                        if (null != activity) {
                            activity.runOnUiThread(() -> Toast.makeText(context,
                                    "could not connect to server, please check connection",
                                    Toast.LENGTH_SHORT).show());
                        } else {
                            Log.println(Log.WARN, "PoclImageProcessor.java:imageProcessLoop",
                                    "could " +
                                            "not connect to server, please check connection");
                        }

                        return;
                    }
                    case 0: {
                        // everything is fine
                        break;
                    }
                    default: {
                        Log.println(Log.ERROR, "PoclImageProcessor.java:imageProcessLoop",
                                "initPoclImageProcessorV2 returned an error" + status);
                        return;
                    }

                }

                // start the thread to read images
                startReceiverThread();

                // the main loop, will continue until an interrupt is sent
                while (!Thread.interrupted()) {

                    // artificial frame limiting
                    if (targetFPS >= 1) {
                        long timeNow = System.nanoTime();
                        long timeSince = (timeNow - lastFrameTime);
                        if (timeSince < (1_000_000_000 / targetFPS)) {
                            Thread.sleep(0, 333333);
                            continue;
                        }
                        lastFrameTime = timeNow;
                    }

                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started new image process" +
                                " iteration");
                    }

                    // wait until image is available,
                    imageAvailableLock.acquire();
                    image = imageReader.acquireLatestImage();

                    // acquirelatestimage closes all other images,
                    // so release all permits related to those images
                    // like this, we will only acquire a lock when a
                    // new image is available
                    int drainedPermits = imageAvailableLock.drainPermits();
                    imageAcquireTime = System.nanoTime();

                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "acquired image");
                    }

                    if (VERBOSITY >= 1) {
                        Log.println(Log.INFO, "imageprocessloop",
                                "drained permits: " + drainedPermits);
                    }

                    if (null == image) {
                        continue;
                    }

                    currentInferencingDevice = inferencingDevice;
                    int sem_status = dequeue_spot(60, currentInferencingDevice);
                    if (sem_status != 0) {
                        if (DEBUGEXECUTION) {
                            Log.println(Log.INFO, "poclimageprocessor", "no spot in pocl queue");
                        }
                        image.close();
                        continue;
                    }

                    rotation = orientationsSwapped ? 90 : 0;
                    do_segment = this.doSegment ? 1 : 0;
                    do_algorithm = enableQualityAlgorithm ? 1 : 0;

                    Image.Plane[] planes = image.getPlanes();

                    // Camera's timestamp passed to the processor to link camera and pocl logs
                    // together
                    long imageTimestamp = image.getTimestamp();

                    // check that the image is supported
                    assert checkImageFormat(image);
                    energy = statLogger.getCurrentEnergy();

                    ByteBuffer Y = planes[0].getBuffer();
                    YPixelStride = planes[0].getPixelStride();
                    YRowStride = planes[0].getRowStride();

                    ByteBuffer U = planes[1].getBuffer();
                    ByteBuffer V = planes[2].getBuffer();
                    UVPixelStride = planes[1].getPixelStride();
                    UVRowStride = planes[1].getRowStride();

                    VPixelStride = planes[2].getPixelStride();
                    VRowStride = planes[2].getRowStride();

                    if (VERBOSITY >= 3) {

                        Log.println(Log.INFO, "imagereader", "image type: " + imageFormat);
                        Log.println(Log.INFO, "imagereader", "plane count: " + planes.length);
                        Log.println(Log.INFO, "imagereader",
                                "Y pixel stride: " + YPixelStride);
                        Log.println(Log.INFO, "imagereader",
                                "Y row stride: " + YRowStride);
                        Log.println(Log.INFO, "imagereader",
                                "UV pixel stride: " + UVPixelStride);
                        Log.println(Log.INFO, "imagereader",
                                "UV row stride: " + UVRowStride);

                        Log.println(Log.INFO, "imagereader",
                                "V pixel stride: " + VPixelStride);
                        Log.println(Log.INFO, "imagereader",
                                "V row stride: " + VRowStride);
                    }

                    currentTime = System.currentTimeMillis();

                    // TODO: Make this selectable
                    boolean measuringIdlePower = false;

                    if (measuringIdlePower) {
                        status = 0;
                    } else {
                        status = poclSubmitYUVImage(
                                currentInferencingDevice, do_segment, compressionType, quality,
                                rotation, do_algorithm,
                                Y, YRowStride, YPixelStride,
                                U, UVRowStride, UVPixelStride,
                                V, VRowStride, VPixelStride,
                                imageTimestamp);
                    }

                    if (status != 0) {
                        throw new IllegalStateException("native poclSubmitYUVImage returned " +
                                "error: " + status);
                    }

                    doneTime = System.currentTimeMillis();
                    poclTime = doneTime - currentTime;
                    if (VERBOSITY >= 2) {
                        Log.println(Log.INFO, "imageprocessloop",
                                "pocl compute time: " + poclTime + "ms");
                    }

                    // don't forget to close the image when done
                    image.close();
                }

                interrupted = true;

            } catch (InterruptedException e) {
                // if an image was open, close it.
                // can be null if the imagereader didn't have an image available
                if (image != null) {
                    image.close();
                }
                Log.println(Log.INFO, "MainActivity.java:imageProcessLoop", "received " +
                        "interrupt, closing down");

                interrupted = true;

            } catch (Exception e) {

                if (image != null) {
                    image.close();
                }
                Log.println(Log.INFO, "MainActivity.java:imageProcessLoop", "error while " +
                        "processing image");
                e.printStackTrace();

            } finally {

                stopReceiverThread();

                // always free pocl
                destroyPoclImageProcessorV2();

                closeParcelFd(logParcelFd);
                closeParcelFd(calibrateParcelFd);

                if (DEBUGEXECUTION) {
                    Log.println(Log.INFO, "EXECUTIONFLOW", "finishing image process loop");
                }
            }

            // if fallback is enabled, start the loop again
            if (ENABLEFALLBACK && !interrupted) {
                runtimeConfigFlags = fallbackToLocal();
                runtimeServiceName = null;
            } else {
                break;
            }

        }

    }

    void closeParcelFd(ParcelFileDescriptor fd) {
        if (null != fd) {
            try {
                fd.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Handle a device being lost.
     *
     * @return new configflags to use during execution
     */
    private int fallbackToLocal() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started fallback to local method");
        }

        if (null != activity) {

            // send it to the main activity
            CodecConfig config = new CodecConfig(NO_COMPRESSION, LOCAL_DEVICE, this.quality,
                    this.quality, 0);
            activity.setButtonsFromJNI(config);
            activity.enableRemote(false);
            activity.runOnUiThread(() -> Toast.makeText(context,
                    "lost connect to remote, local execution only",
                    Toast.LENGTH_SHORT).show());
        }

        this.setCompressionType(NO_COMPRESSION);
        this.setInferencingDevice(LOCAL_DEVICE);
        boolean doProfiling = (configFlags & ENABLE_PROFILING) > 1;
        return NO_COMPRESSION | LOCAL_ONLY | (doProfiling ? ENABLE_PROFILING : 0);
    }

    private void receiveResultLoop() {

        int MAX_DETECTIONS = 10;
        int MASK_SZ1 = 160;
        int MASK_SZ2 = 120;

        int detection_count = 1 + MAX_DETECTIONS * 6;
        int seg_postprocess_count = 4 * MASK_SZ1 * MASK_SZ2;

        int[] detection_results = new int[detection_count];
        byte[] segmentation_results = new byte[seg_postprocess_count];
        // a trick to have pointer like functionality.
        // different indexes point to different values:
        // index 0: do segment
        // 1: latency
        long[] dataExchange = {0, 0};
        float energy = 0;
        int status = 0;

        while (!Thread.interrupted()) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started new image process" +
                        " iteration");
            }

            int sem_status = waitImageAvailable(60);
            if (0 != sem_status) {
                if (DEBUGEXECUTION) {
                    Log.println(Log.INFO, "poclimageprocessor", "no image available pocl queue");
                }
                continue;
            }

            energy = statLogger.getCurrentEnergy();
            status = receiveImage(detection_results, segmentation_results, dataExchange, energy);

            if (status != 0) {
                Log.println(Log.WARN, "poclimageprocessor",
                        "jni receive image returned error: " + status);
                break;
            }


            this.lastIou = poclGetLastIouV2();

            if (null != activity) {

                activity.drawOverlay((int) dataExchange[0], detection_results, segmentation_results,
                        captureSize, orientationsSwapped);

                if (enableQualityAlgorithm) {
                    // TODO: move outside loop so that pip.java also works on non activity objects
                    // Decide which codec to use for the next frame
                    poclSelectCodecAuto();

                    // Fetch the compression type, device, etc., from the JNI and flip the
                    // buttons accordingly
                    CodecConfig config = getCodecConfig();
                    activity.setButtonsFromJNI(config);
                }
            }

            // used to calculate the (avg) FPS
            if (null != counter) {
                counter.tickFrame(dataExchange[1]);
            }

        }
    }

    /**
     * function to start the image process thread, and thereby the
     * main image process loop
     */
    public void start() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startImageProcessThread method");
        }

        imageSubmitThread = new Thread() {
            public void run() {
                imageProcessLoop(null);
            }
        };

        imageSubmitThread.start();

    }

    public void start(String serviceName) {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startImageProcessThread method");
        }

        imageSubmitThread = new Thread() {
            public void run() {
                imageProcessLoop(serviceName);
            }
        };

        imageSubmitThread.start();

    }

    /**
     * function to safely stop the image process thread and
     * ask it nicely to stop anything it is doing
     */
    public void stop() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started stopImageProcessThread method");
        }
        if (null == imageSubmitThread) {
            return;
        }

        // sending an interrupt will exit the while loop
        imageSubmitThread.interrupt();
        try {
            // wait for the last iteration to be done
            imageSubmitThread.join();
            imageSubmitThread = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void startReceiverThread() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startReceiverThread method");
        }

        receiverThread = new Thread() {
            public void run() {
                receiveResultLoop();
            }
        };
        receiverThread.start();
    }

    public void stopReceiverThread() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started stopReceiverThread method");
        }

        if (null == receiverThread) {
            return;
        }

        // sending an interrupt will exit the while loop
        receiverThread.interrupt();
        try {
            // wait for the last iteration to be done
            receiverThread.join();
            receiverThread = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
