/**
 * a foreground service that takes a video as input instead of a camera and processes images.
 */

package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.BENCHMARKVIDEOURI;
import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLECOMPRESSIONKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLELOGGINGKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLESEGMENTATIONKEY;
import static org.portablecl.poclaisademo.BundleKeys.IMAGECAPTUREFRAMETIMEKEY;
import static org.portablecl.poclaisademo.BundleKeys.LOGKEYS;
import static org.portablecl.poclaisademo.BundleKeys.TOTALLOGS;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIutils.setNativeEnv;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class BenchmarkService extends Service {


    /**
     * unique id for the notification
     */
    public static final int NOTIFICATION_ID = 42;

    /**
     * unique id for the notification channel
     */
    public static final String CHANNEL_ID = "BenchmarkserviceChannelID";

    /**
     * description of the benchmark service channel name
     */
    public static final String CHANNEL_NAME = "BenchmarkServiceChannel";

    /**
     * thread to handle callbacks
     */
    private HandlerThread backgroundThread;

    /**
     * used to enqueue work on the background thread
     */
    private Handler backgroundThreadHandler;

    /**
     * the id of service
     */
    private int startId;

    /**
     * java object to manage procssing images
     */
    private PoclImageProcessor poclImageProcessor;

    /**
     * used to open and start the benchmark video
     */
    private MediaPlayer mediaPlayer;

    /**
     * used to read images from the mediaplayer
     */
    private ImageReader imageReader;

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
     * uri to the video to play
     */
    private Uri videoUri;

    /**
     * indicate the size of the images
     */
    private Size captureSize;

    /**
     * the size of the buffer of imagereader
     */
    private int imageBufferSize;

    /**
     * the format used to process
     */
    private int captureFormat;

    /**
     * the index of the device to use
     * 0 : basic
     * 1 : proxy
     * 2 : remote
     * 3 : remote decoder
     */
    private int deviceIndex;

    /**
     * a semaphore used to sync the image process loop with available images.
     * zero starting permits, so a lock can only be acquired when the
     * camerareader releases a permit.
     */
    private final Semaphore imageAvailableLock = new Semaphore(0);

    /**
     * a future to stop the logger
     */
    private ScheduledFuture statLoggerFuture;

    /**
     * used to schedule runnables
     */
    private ScheduledExecutorService schedulerService;

    /**
     * a future to stop the image release runnable
     */
    private ScheduledFuture imageReleaseFuture;

    private int imageFrameTime;

    private int configFlags;

    /**
     * needed to call native pocl functions
     */
    static {
        System.loadLibrary("poclaisademo");
    }

    public BenchmarkService() {

    }

    /**
     * @param intent  The Intent supplied to {@link android.content.Context#startService},
     *                as given.  This may be null if the service is being restarted after
     *                its process has gone away, and it had previously returned anything
     *                except {@link #START_STICKY_COMPATIBILITY}.
     * @param flags   Additional data about this start request.
     * @param startId A unique integer representing this specific request to
     *                start.  Use with {@link #stopSelfResult(int)}.
     * @return
     */
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        this.startId = startId;
        ConfigStore configStore = new ConfigStore(this);

        // todo: read variables set by user from intent
        // get bundle with variables set during startup activity
        Bundle bundle = intent.getExtras();
        String IPAddress = configStore.getIpAddressText();
        boolean disableRemote = bundle.getBoolean(DISABLEREMOTEKEY);
        deviceIndex = disableRemote ? 0 : 2;
        boolean enableSegmentation = bundle.getBoolean(ENABLESEGMENTATIONKEY, true);
        boolean enableCompression = bundle.getBoolean(ENABLECOMPRESSIONKEY, false);
        imageFrameTime = bundle.getInt(IMAGECAPTUREFRAMETIMEKEY, 0);

        boolean enableLogging;
        try {
            enableLogging = bundle.getBoolean(ENABLELOGGINGKEY, false);
        } catch (Exception e) {
            if (VERBOSITY >= 2) {
                Log.println(Log.INFO, "benchmarkService:logging", "could not read enablelogging");
            }
            enableLogging = false;
        }

        for (int i = 0; i < TOTALLOGS; i++) {
            if (enableLogging) {

                try {
                    uris[i] = Uri.parse(bundle.getString(LOGKEYS[i], null));
                } catch (Exception e) {
                    if (VERBOSITY >= 2) {
                        Log.println(Log.INFO, "benchmarkService:logging", "could not parse uri");
                    }
                    uris[i] = null;
                }
            }
            parcelFileDescriptors[i] = null;
            logStreams[i] = null;

        }

        try {
            videoUri = Uri.parse(bundle.getString(BENCHMARKVIDEOURI, null));
        } catch (Exception e) {
            e.printStackTrace();
            videoUri = null;
            stopSelf();
        }

        // related to imagereaders
//        captureSize = new Size(640, 480);
        captureSize = new Size(480, 640);
        imageBufferSize = 35;
        // todo: set the proper format
        captureFormat = ImageFormat.YUV_420_888;

        if (disableRemote) {
            setNativeEnv("POCL_DEVICES", "basic");
        } else {
            setNativeEnv("POCL_DEVICES", "basic remote remote proxy");
            setNativeEnv("POCL_REMOTE0_PARAMETERS", IPAddress + ":10998/0");
            setNativeEnv("POCL_REMOTE1_PARAMETERS", IPAddress + ":10998/1");
        }

        String cache_dir = getCacheDir().getAbsolutePath();
        // used to configure pocl
        setNativeEnv("POCL_DEBUG", "basic,proxy,remote,error");
        setNativeEnv("POCL_CACHE_DIR", cache_dir);

        configFlags = configStore.getConfigFlags();
        boolean enableQualityAlgorithm = configStore.getQualityAlgorithmOption();
        poclImageProcessor = new PoclImageProcessor(this, captureSize, null,
                ImageFormat.YUV_420_888, imageAvailableLock, configFlags, null, deviceIndex,
                enableSegmentation, enableCompression, uris[0], null, enableQualityAlgorithm);
//        poclImageProcessor.setOrientation(true);

        // foreground apps only work on api version 26 and up
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Intent notificationIntent = new Intent(this, BenchmarkConfigurationActivity.class);
            PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                    notificationIntent, PendingIntent.FLAG_IMMUTABLE);

            NotificationChannel notificationChannel = new NotificationChannel(CHANNEL_ID,
                    CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT);
            notificationChannel.setDescription("Benchmark service notification channel");

            NotificationManager notificationManager = getSystemService(NotificationManager.class);
            // it is safe to call this multiple times
            notificationManager.createNotificationChannel(notificationChannel);

            Notification.Builder notificationBuilder = new Notification.Builder(this, CHANNEL_ID);
            notificationBuilder.setContentTitle("Benchmark Service")
                    .setContentText("Benchmark is running")
                    .setSmallIcon(R.drawable.ic_launcher_foreground)
                    .setContentIntent(pendingIntent)
                    .setTicker("ticker text");

            Notification notification = notificationBuilder.build();
            startForeground(NOTIFICATION_ID, notification);

        }

        // log energy and traffic statistics
        EnergyMonitor energyMonitor = new EnergyMonitor(getApplicationContext());
        openFileOutputStream(1);
        StatLogger statLogger = new StatLogger(logStreams[1], new TrafficMonitor(), energyMonitor);
        schedulerService = Executors.newScheduledThreadPool(2);
        statLoggerFuture = schedulerService.scheduleAtFixedRate(statLogger, 1000, 500,
                TimeUnit.MILLISECONDS);
        imageReleaseFuture = null;

        backgroundThread = new HandlerThread("BenchmarkServiceBackgroundThread");
        backgroundThread.start();
        backgroundThreadHandler = new Handler(backgroundThread.getLooper());
        backgroundThreadHandler.post(() -> setupBenchmark());

        // todo: possibly change to sticky
        return START_NOT_STICKY;
    }

    /**
     * create imagereader and mediaplayer and connect objects to each other.
     * This function also starts the playback of the mediaplayer and the poclimageprocessor.
     * These functions may take a while, therefore, run on a different thread.
     */
    private void setupBenchmark() {

        imageReader = ImageReader.newInstance(captureSize.getWidth(), captureSize.getHeight(),
                captureFormat, imageBufferSize);
        Surface imageReaderSurface = imageReader.getSurface();

        // if the frametime is set, use that to release an image lock, which in turn
        // starts a new iteration for the pocl image processor.
        if (0 == imageFrameTime) {
            imageReader.setOnImageAvailableListener(imageAvailableListener,
                    backgroundThreadHandler);
        } else {
            imageReleaseFuture = schedulerService.scheduleAtFixedRate(frameRateRunnable, 1000,
                    imageFrameTime, TimeUnit.MILLISECONDS);
        }

        mediaPlayer = new MediaPlayer();
        mediaPlayer.setOnCompletionListener(onCompletionListener);
        try {
            mediaPlayer.setDataSource(this, videoUri);
            mediaPlayer.setSurface(imageReaderSurface);
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (IOException e) {
            stopSelf();
            throw new RuntimeException(e);
        }

        poclImageProcessor.setImageReader(imageReader);
        poclImageProcessor.start();

    }

    /**
     * Callback to release image semaphore
     */
    private final ImageReader.OnImageAvailableListener imageAvailableListener =
            new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    if (DEBUGEXECUTION && VERBOSITY >= 3) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "image available");
                    }

                    if (0 == imageFrameTime) {
                        return;
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
     * This runnable releases the imageAvailableLock. This is useful to limit the
     * poclImageProcessorLoop to a fixed rate.
     */
    private final Runnable frameRateRunnable = new Runnable() {

        @Override
        public void run() {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "releasing scheduled image lock");
            }
            imageAvailableLock.release();
        }
    };

    /**
     * callback to stop service after video is done playing
     */
    private final MediaPlayer.OnCompletionListener onCompletionListener =
            new MediaPlayer.OnCompletionListener() {

                @Override
                public void onCompletion(MediaPlayer mp) {
                    stopSelf(startId);
                }
            };

    /**
     * @param intent The Intent that was used to bind to this service,
     *               as given to {@link android.content.Context#bindService
     *               Context.bindService}.  Note that any extras that were included with
     *               the Intent at that point will <em>not</em> be seen here.
     * @return
     * @note not used
     */
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    /**
     * Gets called as part of stopSelf(). Stops mediaplayer, closes imagereader, join threads
     * and stops poclimageprocessor.
     */
    @Override
    public void onDestroy() {
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "BenchmarkService", "destroying service");
        }

        mediaPlayer.setSurface(null);
        mediaPlayer.stop();
        mediaPlayer.reset();

        if (null != mediaPlayer) {
            mediaPlayer.release();
        }

        poclImageProcessor.stop();

        // there might be some warnings like:
        // detachBuffer: slot 0 is not owned by the producer (state = FREE)
        // but this seems to be benign, see:
        // https://android.googlesource.com/platform/frameworks/native/+/master/libs/gui/BufferQueueProducer.cpp
        if (null != imageReader) {
            imageReader.close();
        }

        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        statLoggerFuture.cancel(true);
        // unlike the statlogger, this future is optional.
        if (null != imageReleaseFuture) {
            imageReleaseFuture.cancel(true);
        }

        closeFileOutputStreams();

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "BenchmarkService", "done destroying service");
        }

        super.onDestroy();
    }

    /**
     * open all the log files for writing
     *
     * @return true if successful
     */
    private boolean openFileOutputStream(int i) {
        try {
            parcelFileDescriptors[i] = getContentResolver().openFileDescriptor(uris[i], "wa");
            logStreams[i] = new FileOutputStream(parcelFileDescriptors[i].getFileDescriptor());

        } catch (Exception e) {
            Log.println(Log.WARN, "openFileOutputStreams", "could not open log file " + i);
            logStreams[i] = null;
            return false;
        }
        return true;
    }

    /**
     * close all the log files
     */
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

}