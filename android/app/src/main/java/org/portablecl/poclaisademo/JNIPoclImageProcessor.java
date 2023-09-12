/**
 *  JNIPoclImageProcessor.java a file with java native interfaces implemented by
 *  poclImageProcessor.cpp
 */
package org.portablecl.poclaisademo;
import android.content.res.AssetManager;

import java.io.FileDescriptor;
import java.nio.ByteBuffer;

public class JNIPoclImageProcessor {

    public final static int NO_COMPRESSION = 1;
    public final static int YUV_COMPRESSION = 2;
    public final static int JPEG_COMPRESSION = 4;
    public final static int ENABLE_PROFILING = (1 << 8);

    public final static int LOCAL_DEVICE = 0;

    public final static int PASSTHRU_DEVICE = 1;
    public final static int REMOTE_DEVICE = 2;

    public static native int initPoclImageProcessor(int configFlags,
                                                    AssetManager jAssetManager,
                                                    int width, int height, int fd);

    public static native int destroyPoclImageProcessor();

    public static native int poclProcessYUVImage(int deviceIndex, int do_segment,
                                                 int doCompression, int quality, int rotation,
                                                 ByteBuffer Y, int YRowStride, int YPixelStride,
                                                 ByteBuffer U, ByteBuffer V, int UVRowStride,
                                                 int UVPixelStride, int[] detection_result,
                                                 byte[] segmentation_result);

    public static native String getProfilingStats();

    public static native String getCSVHeader();

    public static native byte[] getProfilingStatsbytes();

    public static native TrafficMonitor.DataPoint getRemoteTrafficStats();

}
