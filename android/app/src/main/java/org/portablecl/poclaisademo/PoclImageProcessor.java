package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.destroyPoclImageProcessor;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.getPrfilingStatsbytes;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.initPoclImageProcessor;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.poclProcessYUVImage;

import android.content.Context;
import android.content.res.AssetManager;
import android.media.Image;
import android.media.ImageReader;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import java.io.FileOutputStream;
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
    private Thread imageProcessThread;

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

    /**
     * used to indicate if logging is enabled
     */
    private final boolean enableLogging;

    /**
     * used to write to logging file
     */
    private final FileOutputStream[] logStreams;

    private boolean orientationsSwapped;

    private boolean doSegment;

    private boolean doCompression;

    public int inferencingDevice;


    /**
     * constructor for pocl image processor
     *
     * @param context
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param enableLogging
     * @param logStreams
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     * @param doCompression
     */
    public PoclImageProcessor(Context context, Size captureSize, ImageReader imageReader,
                              Semaphore imageAvailableLock, boolean enableLogging,
                              FileOutputStream[] logStreams, FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment, boolean doCompression) {
        this(null, context, captureSize, imageReader, imageAvailableLock, enableLogging,
                logStreams,
                fpsCounter, inferencingDevice, doSegment, doCompression);
    }

    /**
     * constructor for pocl image processor
     *
     * @param activity
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param enableLogging
     * @param logStreams
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     * @param doCompression
     */
    public PoclImageProcessor(MainActivity activity, Size captureSize, ImageReader imageReader,
                              Semaphore imageAvailableLock, boolean enableLogging,
                              FileOutputStream[] logStreams, FPSCounter fpsCounter,
                              int inferencingDevice, boolean doSegment, boolean doCompression) {
        this(activity, activity, captureSize, imageReader, imageAvailableLock, enableLogging,
                logStreams,
                fpsCounter, inferencingDevice, doSegment, doCompression);
    }

    /**
     * Constructor of the pocl image processor
     *
     * @param activity
     * @param context
     * @param captureSize
     * @param imageReader
     * @param imageAvailableLock
     * @param enableLogging
     * @param logStreams
     * @param fpsCounter
     * @param inferencingDevice
     * @param doSegment
     * @param doCompression
     */
    private PoclImageProcessor(MainActivity activity, Context context, Size captureSize,
                               ImageReader imageReader,
                               Semaphore imageAvailableLock, boolean enableLogging,
                               FileOutputStream[] logStreams, FPSCounter fpsCounter,
                               int inferencingDevice, boolean doSegment, boolean doCompression) {
        this.activity = activity;
        this.context = context;
        this.captureSize = captureSize;
        this.imageReader = imageReader;
        this.imageAvailableLock = imageAvailableLock;
        this.enableLogging = enableLogging;
        this.logStreams = logStreams;

        counter = fpsCounter;
        this.inferencingDevice = inferencingDevice;
        this.doSegment = doSegment;
        this.doCompression = doCompression;
        this.orientationsSwapped = false;
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

    /**
     * Set the imageReader to use
     *
     * @param imageReader
     */
    public void setImageReader(ImageReader imageReader) {
        this.imageReader = imageReader;
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
        int seg_postprocess_count = 4 * MASK_W * MASK_H;

        int[] detection_results = new int[detection_count];
        byte[] segmentation_results = new byte[seg_postprocess_count];

        int YPixelStride, YRowStride;
        int UVPixelStride, UVRowStride;
        int VPixelStride, VRowStride;
        int rotation, do_segment, do_compression;
        long currentTime, doneTime, poclTime;

        Image image = null;
        StringBuilder logBuilder = new StringBuilder();
        if (enableLogging) {
            try {
                logStreams[0].write(("start_time, stop_time, inferencing_device, do_segment, " +
                        "do_compression, image_to_buffer, run_yolo, read_detections, " +
                        "run_postprocess, read_segments, run_enc_y, run_enc_uv, run_dec_y, " +
                        "run_dec_uv \n").getBytes());
            } catch (IOException e) {
                Log.println(Log.WARN, "Mainactivity.java:imageProcessLoop", "could not write csv " +
                        "header");
            }
        }
        try {
            AssetManager assetManager = context.getAssets();
            int status = initPoclImageProcessor(enableLogging, assetManager,
                    captureSize.getWidth(),
                    captureSize.getHeight());

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

            // the main loop, will continue until an interrupt is sent
            while (!Thread.interrupted()) {

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

                Image.Plane[] planes = image.getPlanes();

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

                rotation = orientationsSwapped ? 90 : 0;
                do_segment = doSegment ? 1 : 0;
                do_compression = doCompression ? 1 : 0;

                currentTime = System.currentTimeMillis();
                poclProcessYUVImage(inferencingDevice, do_segment, do_compression,
                        rotation, Y, YRowStride, YPixelStride, U, V, UVRowStride, UVPixelStride,
                        detection_results, segmentation_results);
                doneTime = System.currentTimeMillis();
                poclTime = doneTime - currentTime;
                if (VERBOSITY >= 1) {
                    Log.println(Log.WARN, "imageprocessloop",
                            "pocl compute time: " + poclTime + "ms");
                }

                if (null != activity) {
                    activity.drawOverlay(do_segment, detection_results, segmentation_results,
                            captureSize, orientationsSwapped);
                }


                if (enableLogging) {
                    // clear the stringbuilder
                    logBuilder.setLength(0);
                    logBuilder.append(currentTime).append(", ").append(doneTime).append(", ");
                    logStreams[0].write(logBuilder.toString().getBytes());
                    logStreams[0].write(getPrfilingStatsbytes());

                }

                // used to calculate the (avg) FPS
                if(null != counter){
                    counter.tickFrame();
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
            // always free pocl
            destroyPoclImageProcessor();
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "finishing image process loop");
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
    public void stop() {
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
