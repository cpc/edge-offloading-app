package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Locale;

/**
 * A class that is used to log camera related activites. Can be used as a callback
 * to capturesesssion requests
 */
public class CameraLogger extends CameraCaptureSession.CaptureCallback {

    StringBuilder builder;
    FileOutputStream stream;

    public CameraLogger(FileOutputStream stream) {
        this.builder = new StringBuilder();
        this.stream = stream;

        try {
            stream.write("frameindex,tag,sys_ts_ns,dev_ts_ns\n".getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * function to log when an image is consumed for processing
     * <p>
     * Called from poclimageprocessor.java right after the latest image is acquired
     *
     * @param systemTime the time in ns when the image is acquired
     * @param deviceTime the timestamp of the image
     */
    public void logImage(long systemTime, long deviceTime) {
        if (VERBOSITY >= 2) {
            Log.println(Log.WARN, "cameracapture", String.format(Locale.US, "log image sys ts: %d," +
                    " dev ts: %d", systemTime, deviceTime));
        }

        builder.setLength(0);
        builder.append("0,image,").append(systemTime).append(",").append(deviceTime).append(
                "\n");

        try {
            stream.write(builder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Override
    public void onCaptureStarted(@NonNull CameraCaptureSession session,
                                 @NonNull CaptureRequest request, long timestamp,
                                 long frameNumber) {
        long startSystemTime = System.nanoTime();
        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "started  capture sys ts: " +
                    "%16d, frame: %d", startSystemTime, frameNumber));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "started  capture dev " +
                    "ts: %16d, frame: %d", timestamp, frameNumber));
        }

        builder.setLength(0);
        builder.append(frameNumber).append(",camera_start,").append(startSystemTime).append(",").append(timestamp).append("\n");
        try {
            stream.write(builder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                   @NonNull CaptureRequest request,
                                   @NonNull TotalCaptureResult result) {
        long systemTime = System.nanoTime();
        long frameNumber = result.getFrameNumber();
        long devTime = result.get(TotalCaptureResult.SENSOR_TIMESTAMP);
        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "finished capture sys ts:" +
                            " %16d, frame: %d",
                    System.nanoTime(), frameNumber));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "finished capture " +
                    "dev ts: %16d, frame: %d", devTime, frameNumber));
        }

        builder.setLength(0);
        builder.append(frameNumber).append(",camera_finish,").append(systemTime).append(",").append(devTime).append("\n");
        try {
            stream.write(builder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


}
