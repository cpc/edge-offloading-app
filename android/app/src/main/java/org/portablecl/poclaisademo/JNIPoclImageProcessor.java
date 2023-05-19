/**
 *  JNIPoclImageProcessor.java a file with java native interfaces implemented by
 *  poclImageProcessor.cpp
 */
package org.portablecl.poclaisademo;
import android.content.res.AssetManager;

import java.nio.ByteBuffer;

public class JNIPoclImageProcessor {

    public static native int initPoclImageProcessor(AssetManager jAssetManager);

    public static native int destroyPoclImageProcessor();

    public static native int poclProcessYUVImage(int width, int height,
                                                 ByteBuffer Y, int YRowStride, int YPixelStride,
                                                 ByteBuffer U, ByteBuffer V, int UVRowStride,
                                                 int UVPixelStride, int[] result);


}
