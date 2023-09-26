package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.annotation.NonNull;

/**
 * A class that is used to store and share config variables that stay persistent across reboots.
 */
public class ConfigStore {

    private SharedPreferences preferences;

    private SharedPreferences.Editor editor;

    private final static String logTag = "configStore";
    private final static String keyPrefix = "org.portablecl.poclaisademo.";
    private final static String preferencStoreName = keyPrefix + "configstore";

    private final static String configFlagKey = keyPrefix + "configflagkey";

    private final static String uriLogKey = keyPrefix + "urilogkey";

    private final static String jpegQualityKey = keyPrefix + "jpegqualitykey";

    /**
     * @param context can be an activity for example
     */
    public ConfigStore(@NonNull Context context) {
        this.preferences = context.getSharedPreferences(preferencStoreName, Context.MODE_PRIVATE);
        this.editor = preferences.edit();
    }

    /**
     *
     * @return an int with configs encoded on each bit.
     */
    public int getConfigFlags() {

        return preferences.getInt(configFlagKey, NO_COMPRESSION | YUV_COMPRESSION);
    }

    /**
     * add the configflag to be set
     * @param configFlags to be stored
     */
    public void setConfigFlags(int configFlags) {

        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, logTag, String.format("setting config flags to: 0x%08X", configFlags));
        }
        editor.putInt(configFlagKey, configFlags);
    }

    public String getLogUri(String key) {
        return preferences.getString(uriLogKey + key, null);
    }

    public void setLogUri(String key, String value) {
        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, logTag, String.format("setting uri key %s to: %s", key, value));
        }

        editor.putString(uriLogKey + key, value);
    }

    /**
     * get the stored jpeg quality setting. defaults to 80.
     * @return
     */
    public int getJpegQuality() {
        return preferences.getInt(jpegQualityKey, 80);
    }

    /**
     * set the jpeg quality to be stored
     * @param quality to be set, there are no checks if the value is in an acceptable range
     */
    public void setJpegQuality(int quality) {
        if (VERBOSITY >= 2) {
            Log.println(Log.INFO, logTag, String.format("setting jpeg quality to: %d", quality));
        }

        editor.putInt(jpegQualityKey, quality);
    }

    /**
     * write the changes set back so that effects are non volatile.
     */
    public void flushSetting() {
        editor.apply();
    }
}
