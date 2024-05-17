/**
 * JNIPoclImageProcessor.java a file with java native interfaces implemented by
 * poclImageProcessor.cpp
 */
package org.portablecl.poclaisademo;

import android.content.res.AssetManager;

import java.nio.ByteBuffer;

public class JNIPoclImageProcessor {
    // The below constants must match those in poclImageProcessorTypes.h:

    public final static int NO_COMPRESSION = 1;
    public final static int YUV_COMPRESSION = 2;
    public final static int JPEG_COMPRESSION = 4;
    public final static int JPEG_IMAGE = (1 << 3);
    public final static int HEVC_COMPRESSION = (1 << 4);
    public final static int SOFTWARE_HEVC_COMPRESSION = (1 << 5);
    public final static int ENABLE_PROFILING = (1 << 8);

    public final static int LOCAL_DEVICE = 0;
    public final static int PASSTHRU_DEVICE = 1;
    public final static int REMOTE_DEVICE = 2;
    public final static int REMOTE_DEVICE_2 = 3;

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

    public static native int initPoclImageProcessorV2(int configFlags,
                                                      AssetManager jAssetManager,
                                                      int width, int height, int fd, int max_lanes);

    public static native int destroyPoclImageProcessorV2();

    public static native float poclGetLastIouV2();

    public static native int dequeue_spot(int timeout, int dev_type);

    public static native int poclSelectCodecAuto(int doSegment, int rotation);

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

}
