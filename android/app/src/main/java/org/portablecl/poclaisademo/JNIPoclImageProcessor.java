/**
 * JNIPoclImageProcessor.java a file with java native interfaces implemented by
 * poclImageProcessor.cpp
 */
package org.portablecl.poclaisademo;

import android.content.res.AssetManager;
import android.util.Log;

import java.nio.ByteBuffer;
import java.util.HashMap;

public class JNIPoclImageProcessor {
    // The below constants must match those in poclImageProcessorTypes.h:

    public final static int NO_COMPRESSION = 1;
    public final static int YUV_COMPRESSION = 2;
    public final static int JPEG_COMPRESSION = 4;
    public final static int JPEG_IMAGE = (1 << 3);
    public final static int HEVC_COMPRESSION = (1 << 4);
    public final static int SOFTWARE_HEVC_COMPRESSION = (1 << 5);
    /**
     * hashmap that maps string representations of compressionoptions to the number
     */
    public final static HashMap<String, Integer> allCompressionOptions =
            populateCompressionOptions();
    public final static int ENABLE_PROFILING = (1 << 8);
    public final static int LOCAL_ONLY = (1 << 9);
    public final static int LOCAL_DEVICE = 0;
    public final static int PASSTHRU_DEVICE = 1;
    public final static int REMOTE_DEVICE = 2;
    public final static int REMOTE_DEVICE_2 = 3;

    public final static int SEGMENT_4B = (1 << 12);
    public final static int SEGMENT_RLE = (1 << 13);

    /**
     * function that maps a compression option to its string representation
     *
     * @param compressionType to be mapped
     * @return string representation of compression option
     */
    public static String getCompressionString(int compressionType) {

        String returnValue;
        switch (compressionType) {
            case NO_COMPRESSION:
                returnValue = "no compression";
                break;
            case YUV_COMPRESSION:
                returnValue = "YUV";
                break;
            case JPEG_COMPRESSION:
                returnValue = "JPEG";
                break;
            case JPEG_IMAGE:
                returnValue = "JPEG IMAGE";
                break;
            case HEVC_COMPRESSION:
                returnValue = "HEVC";
                break;
            case SOFTWARE_HEVC_COMPRESSION:
                returnValue = "SOFT HEVC";
                break;
            default:
                Log.println(Log.ERROR, "getCompressionString",
                        "unknown compression type: " + compressionType);
                returnValue = "no compression";
                break;
        }
        return returnValue;
    }

    /**
     * function that creates a hashmap of all compression options.
     *
     * @return
     */
    private static HashMap<String, Integer> populateCompressionOptions() {
        HashMap<String, Integer> returnMap = new HashMap<String, Integer>();
        returnMap.put("no compression", NO_COMPRESSION);
        returnMap.put("YUV", YUV_COMPRESSION);
        returnMap.put("JPEG", JPEG_COMPRESSION);
        returnMap.put("JPEG IMAGE", JPEG_IMAGE);
        returnMap.put("HEVC", HEVC_COMPRESSION);
        returnMap.put("SOFT HEVC", SOFTWARE_HEVC_COMPRESSION);
        return returnMap;
    }

    public static native int initPoclImageProcessor(int configFlags,
                                                    AssetManager jAssetManager,
                                                    int width, int height, int fd);

    public static native int destroyPoclImageProcessor();

    public static native float poclGetLastIou();

    public static native int poclProcessYUVImage(int deviceIndex, int doSegment,
                                                 int compressionType,
                                                 int quality, int rotation, int doAlgorithm,
                                                 int[] detectionResult,
                                                 byte[] segmentationResult,
                                                 ByteBuffer plane0, int rowStride0,
                                                 int pixelStride0,
                                                 ByteBuffer plane1, int rowStride1,
                                                 int pixelStride1,
                                                 ByteBuffer plane2, int rowStride2,
                                                 int pixelStride2,
                                                 long imageTimestamp, float energy);

    public static native int poclProcessJPEGImage(int deviceIndex, int doSegment, int doCompression,
                                                  int quality, int rotation, int[] detectionResult,
                                                  byte[] segmentationResult, ByteBuffer data,
                                                  int size, long imageTimestamp, float energy);

    public static native int initPoclImageProcessorV2(int configFlags, AssetManager jAssetManager,
                                                      int width, int height, int fd, int max_lanes,
                                                      int doAlgorithm, int runtimeEval,
                                                      int lockCodec, String serviceName,
                                                      int calibrate);

    public static native int destroyPoclImageProcessorV2();

    public static native float poclGetLastIouV2();

    public static native int dequeue_spot(int timeout, int dev_type);

    public static native int poclSelectCodecAuto();

    public static native int poclSubmitYUVImage(int deviceIndex, int doSegment, int compressionType,
                                                int quality, int rotation, int doAlgorithm,
                                                ByteBuffer plane0, int rowStride0,
                                                int pixelStride0,
                                                ByteBuffer plane1, int rowStride1,
                                                int pixelStride1,
                                                ByteBuffer plane2, int rowStride2,
                                                int pixelStride2,
                                                long imageTimestamp);

    public static native int waitImageAvailable(int timeout);

    public static native int receiveImage(int[] detectionResult, byte[] segmentationResult,
                                          long[] dataExchangeArray,
                                          float energy);

    public static native String getProfilingStats();

    public static native String getCSVHeader();

    public static native byte[] getProfilingStatsbytes();

    public static native TrafficMonitor.DataPoint getRemoteTrafficStats();

    public static native CodecConfig getCodecConfig();
//    public static native MainActivity.Stats getStats();

    // push stats like voltage and current from Java to the C side
    public static native void pushExternalPow(long timestamp, int amp, int volt);

    public static native void pushExternalPing(long timestamp, float ping_ms);
}
