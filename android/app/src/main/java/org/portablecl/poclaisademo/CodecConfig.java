package org.portablecl.poclaisademo;

/**
 * Information about how images should be encoded.
 *
 * The intent is to store parameters from codec_config_t that we need on Java side.
 */
public class CodecConfig {
    public final int compressionType;
    public final int deviceIndex;

    // Not part of codec_config_t, used for debugging
    public final int configIndex;

    public CodecConfig(int compressionType, int deviceIndex, int configIndex) {
        this.compressionType = compressionType;
        this.deviceIndex = deviceIndex;
        this.configIndex = configIndex;
    }
}
