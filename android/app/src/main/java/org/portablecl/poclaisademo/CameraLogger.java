package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
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
            Log.println(Log.WARN, "cameracapture", String.format(Locale.US,
                    "log image sys ts: %d, dev ts: %d", systemTime, deviceTime));
        }

        builder.setLength(0);
        builder.append("0,image,").append(systemTime).append(",").append(deviceTime).append("\n");

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
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d, started  "
                    + "capture sys ts: %16d", frameNumber, startSystemTime));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d, started  "
                    + "capture dev ts: %16d", frameNumber, timestamp));
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
    public void onCaptureProgressed(@NonNull CameraCaptureSession session,
                                    @NonNull CaptureRequest request,
                                    @NonNull CaptureResult partialResult) {
        long systemTime = System.nanoTime();
        long frameNumber = partialResult.getFrameNumber();

        if (VERBOSITY >= 4) {
            for (CaptureResult.Key key : partialResult.getKeys()) {
                Object res = partialResult.get(key);
                if (res.getClass() == Integer.class) {
                    Log.println(Log.INFO, "cameracapture", String.format(Locale.US,
                            "frame: %d, partial sys ts: %16d | %s: %d", frameNumber,
                            systemTime, key.getName(), res));
                } else if (res.getClass() == Float.class) {
                    Log.println(Log.INFO, "cameracapture", String.format(Locale.US,
                            "frame: %d, partial sys ts: %16d | %s: %f", frameNumber,
                            systemTime, key.getName(), res));
                } else {
                    Log.println(Log.INFO, "cameracapture", String.format(Locale.US,
                            "frame: %d, partial sys ts: %16d | %s: %s", frameNumber,
                            systemTime, key.getName(), res));
                }
            }
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
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d, finished "
                    + "capture sys ts: %16d", frameNumber, systemTime));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d, finished "
                    + "capture dev ts: %16d", frameNumber, devTime));
        }

        if (VERBOSITY >= 4) {
            float aperture = result.get(TotalCaptureResult.LENS_APERTURE);
            long exposure_time = result.get(TotalCaptureResult.SENSOR_EXPOSURE_TIME);
            long frame_duration = result.get(TotalCaptureResult.SENSOR_FRAME_DURATION);
            long sensitivity = result.get(TotalCaptureResult.SENSOR_SENSITIVITY);
            byte camera_jpeg_quality = result.get(TotalCaptureResult.JPEG_QUALITY);

            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d,          "
                    + "      aperture: %f", frameNumber, aperture));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d,          "
                    + " exposure time: %16d", frameNumber, exposure_time));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d,          "
                    + "frame duration: %16d", frameNumber, frame_duration));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d,          "
                    + "   sensitivity: %16d", frameNumber, sensitivity));
            Log.println(Log.INFO, "cameracapture", String.format(Locale.US, "frame: %d,          "
                    + "  jpeg quality: %16d", frameNumber, camera_jpeg_quality));
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
