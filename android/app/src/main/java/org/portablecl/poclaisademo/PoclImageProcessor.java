package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.ENABLE_PROFILING;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.SOFTWARE_HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.dequeue_spot;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.destroyPoclImageProcessorV2;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.initPoclImageProcessorV2;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclGetLastIouV2;
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

    /**
     * the image reader is used to get images for processing
     */
    private ImageReader imageReader;

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
     * a thread to run pocl on
     */
    private Thread imageSubmitThread;

    private Thread receiverThread;

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

    private boolean orientationsSwapped;

    private boolean doSegment;

    private boolean doCompression;

    private int compressionType;

    public int inferencingDevice;

    private int quality;

    private final int configFlags;

    private int imageFormat;

    private final StatLogger statLogger;

    private float lastIou = -4.0f;

    private final boolean enableQualityAlgorithm;

    private int targetFPS;

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
     * @param doCompression
     */
    public PoclImageProcessor(Context context, Size captureSize, ImageReader imageReader,
                              int imageFormat, Semaphore imageAvailableLock,
                              int configFlags,
                              FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment, boolean doCompression,
                              Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm) {
        this(null, context, captureSize, imageReader, imageFormat, imageAvailableLock,
                configFlags,
                fpsCounter, inferencingDevice, doSegment, doCompression, uri, statLogger,
                enableQualityAlgorithm);
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
     * @param doCompression
     */
    public PoclImageProcessor(MainActivity activity, Size captureSize, ImageReader imageReader,
                              int imageFormat,
                              Semaphore imageAvailableLock, int configFlags,
                              FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment, boolean doCompression,
                              Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm) {
        this(activity, activity, captureSize, imageReader, imageFormat, imageAvailableLock,
                configFlags,
                fpsCounter, inferencingDevice, doSegment, doCompression, uri, statLogger,
                enableQualityAlgorithm);
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
     * @param doCompression
     */
    private PoclImageProcessor(MainActivity activity, Context context, Size captureSize,
                               ImageReader imageReader, int imageFormat,
                               Semaphore imageAvailableLock, int configFlags,
                               FPSCounter fpsCounter,
                               int inferencingDevice, boolean doSegment, boolean doCompression,
                               Uri uri, StatLogger statLogger, boolean enableQualityAlgorithm) {
        this.activity = activity;
        this.context = context;
        this.captureSize = captureSize;
        this.imageReader = imageReader;
        this.imageAvailableLock = imageAvailableLock;
//        this.enableLogging = enableLogging;
        this.statLogger = statLogger;
        this.enableQualityAlgorithm = enableQualityAlgorithm;

        counter = fpsCounter;
        this.inferencingDevice = inferencingDevice;
        this.doSegment = doSegment;
        this.doCompression = doCompression;
        this.orientationsSwapped = false;
        this.uri = uri;

        this.configFlags = configFlags;
        // default value
        this.quality = 80;

        this.compressionType = NO_COMPRESSION;

        setImageFormat(imageFormat);

        this.imageSubmitThread = null;
        this.receiverThread = null;

        this.targetFPS = 30;

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

    /**
     * Enable compression
     *
     * @param doCompression
     */
    public void setDoCompression(boolean doCompression) {
        this.doCompression = doCompression;
    }

    public void setCompressionType(int compressionType) {
        this.compressionType = compressionType;
    }

    public void setTargetFPS(int targetFPS) {
        if(targetFPS > 30) {
            Log.println(Log.WARN, "PoclImageProcessor.java", "higher target than allowed, capping to 30");
        }
        this.targetFPS = targetFPS;
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
    private void imageProcessLoop() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started image process loop");
        }

        int YPixelStride, YRowStride;
        int UVPixelStride, UVRowStride;
        int VPixelStride, VRowStride;
        int rotation, do_segment, compressionParam;
        long currentTime, doneTime, poclTime, imageAcquireTime;
        int size;
        float energy;
        int currentInferencingDevice; // inference device is needed both for
        // dequeue_spot and submit image, so copy value over to prevent the value changing

        Image image = null;

        ParcelFileDescriptor parcelFileDescriptor = null;

        // used for frame limiting.
        // set a starting time that is definitely smaller
        // than the first time we check this value
        long lastFrameTime = System.nanoTime() - 10_000_000_000L;

        int nativeFd = -1;
        if ((ENABLE_PROFILING & configFlags) > 0) {
            try {
                parcelFileDescriptor = context.getContentResolver().openFileDescriptor(uri, "wa");
            } catch (FileNotFoundException e) {
                Log.println(Log.WARN, "PoclImageProcessor.java:imageProcessLoop",
                        "could not open log filedescriptor");
                return;
            }
            nativeFd = parcelFileDescriptor.getFd();
        }
        try {
            AssetManager assetManager = context.getAssets();
            int status = initPoclImageProcessorV2(configFlags, assetManager,
                    captureSize.getWidth(),
                    captureSize.getHeight(), nativeFd, 2);

            if (-33 == status) {
                if (null != activity) {
                    activity.runOnUiThread(() -> Toast.makeText(context,
                            "could not connect to server, please check connection",
                            Toast.LENGTH_SHORT).show());
                } else {
                    Log.println(Log.WARN, "PoclImageProcessor.java:imageProcessLoop", "could " +
                            "not connect to server, please check connection");
                }

                return;
            }

            // start the thread to read images
            startReceiverThread();

            // the main loop, will continue until an interrupt is sent
            while (!Thread.interrupted()) {

                // artificial frame limiting
                if(targetFPS >= 1) {
                    long timeNow = System.nanoTime();
                    long timeSince = (timeNow- lastFrameTime);
                    if(timeSince < (1_000_000_000 / targetFPS)){
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

                Image.Plane[] planes = image.getPlanes();
                rotation = orientationsSwapped ? 90 : 0;
                do_segment = doSegment ? 1 : 0;
                // pick the right compression value
                compressionParam = doCompression ? compressionType : NO_COMPRESSION;

                // Camera's timestamp passed to the processor to link camera and pocl logs together
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

                int doAlgorithm = enableQualityAlgorithm ? 1 : 0;
                currentTime = System.currentTimeMillis();

                poclSubmitYUVImage(currentInferencingDevice, do_segment, compressionParam, quality,
                        rotation, doAlgorithm,
                        Y, YRowStride, YPixelStride,
                        U, UVRowStride, UVPixelStride,
                        V, VRowStride, VPixelStride,
                        imageTimestamp);

                doneTime = System.currentTimeMillis();
                poclTime = doneTime - currentTime;
                if (VERBOSITY >= 2) {
                    Log.println(Log.INFO, "imageprocessloop",
                            "pocl compute time: " + poclTime + "ms");
                }

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

            stopReceiverThread();

            // always free pocl
            destroyPoclImageProcessorV2();

            if (null != parcelFileDescriptor) {
                try {
                    parcelFileDescriptor.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "finishing image process loop");
            }
        }

    }

    private void receiveResultLoop() {

        int MAX_DETECTIONS = 10;
        int MASK_W = 160;
        int MASK_H = 120;

        int detection_count = 1 + MAX_DETECTIONS * 6;
        int seg_postprocess_count = 4 * MASK_W * MASK_H;

        int[] detection_results = new int[detection_count];
        byte[] segmentation_results = new byte[seg_postprocess_count];
        // a trick to have pointer like functionality.
        // different indexes point to different values:
        // index 0: do segment
        // 1: latency
        long[] dataExchange = {0, 0};
        float energy = 0;

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
            receiveImage(detection_results, segmentation_results, dataExchange, energy);

            this.lastIou = poclGetLastIouV2();

            if (null != activity) {
                activity.drawOverlay((int) dataExchange[0], detection_results, segmentation_results,
                        captureSize, orientationsSwapped);

                // update the buttons
                if (enableQualityAlgorithm) {
                    activity.setButtonsFromJNI();
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
                imageProcessLoop();
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
