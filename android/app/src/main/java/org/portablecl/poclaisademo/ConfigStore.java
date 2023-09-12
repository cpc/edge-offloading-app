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

    private final String logTag = "configStore";
    private final String keyPrefix = "org.portablecl.poclaisademo.";
    private final String preferencStoreName = keyPrefix +"configstore";

    private final String configFlagKey = keyPrefix + "configflagkey";

    /**
     *
     * @param context can be an activity for example
     */
    public ConfigStore(@NonNull Context context){
        this.preferences = context.getSharedPreferences(preferencStoreName,Context.MODE_PRIVATE);
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

    /**
     * write the changes set back so that effects are non volatile.
     */
    public void flushSetting() {
        editor.apply();
    }
}
