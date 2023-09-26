package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.CAMERALOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLELOGGINGKEY;
import static org.portablecl.poclaisademo.BundleKeys.IPKEY;
import static org.portablecl.poclaisademo.BundleKeys.MONITORLOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.POCLLOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.TOTALLOGS;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.ENABLE_PROFILING;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;
import static java.lang.Character.isDigit;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.portablecl.poclaisademo.databinding.ActivityStartupBinding;


public class StartupActivity extends AppCompatActivity {

    private static final String[] preferencekeys = {"org.portablecl.poclaisademo.logfile.urikey",
            "org.portablecl.poclaisademo.logfile.monitorurikey", "org.portablecl.poclaisademo.logfile.cameralogurikey"};

    private static final String[] filePrefixes = {"pocl", "monitor", "camera"};

    /**
     * static prefix to use with displaying the file name
     */
    private static final String fileTextPrefix = "file name: ";

    /**
     * textview where the user inputs the ip address
     */
    private AutoCompleteTextView IPAddressView;

    /**
     * used to display the name of the file chosen
     */
    private final TextView[] fileViews = new TextView[TOTALLOGS];

    /**
     * suggestion ip addresses
     */
    private final static String[] IPAddresses = {"192.168.36.206", "192.168.50.112", "10.1.200.5"};

    /**
     * a boolean that is passed to the main activity disable remote
     */
    private boolean disableRemote;

    /**
     * Universal Resource Identifier (URI) used to point the log file.
     * In newer android versions, you can not point to a file with paths.
     * Instead, you have to get/generate an URI which points to it.
     */
    private final Uri[] uris = new Uri[TOTALLOGS];

    /**
     * boolean to set when a file is found
     */
    private final boolean[] fileExistsChecks = new boolean[TOTALLOGS];

    /**
     * boolean to store user values
     */
    private boolean enableLogging;

    /**
     * Quality parameter of camera's JPEG compression
     */
    private int jpegQuality;

    private Switch enableLoggingSwitch;

    /**
     * value store that persists across app life cycles
     */
    private SharedPreferences sharedPreferences;

    private ConfigStore configStore;

    private ToggleButton yuvCompButton;
    private ToggleButton jpegCompButton;

    private ToggleButton jpegImageButton;

    private Switch cameraLogSwitch;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityStartupBinding binding = ActivityStartupBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        configStore = new ConfigStore(this);

        // load uri from app preferences. This keeps the data between, app restarts.
        //https://developer.android.com/training/data-storage/shared-preferences
        sharedPreferences = getPreferences(Context.MODE_PRIVATE);
        for (int i = 0; i < TOTALLOGS; i++) {
            try {
                uris[i] = Uri.parse(sharedPreferences.getString(preferencekeys[i], null));
            } catch (Exception e) {
                Log.println(Log.INFO, "startupactivity",
                        "could not parse stored" + preferencekeys[i] + " uri");
                uris[i] = null;
            }
        }

        // display file name
        fileViews[0] = binding.poclFileNameView;
        fileViews[1] = binding.monitorFileNameView;
        fileViews[2] = binding.cameraLogFileNameView;

        for(int i = 0; i < TOTALLOGS; i++) {
            String fileName = getFileName(uris[i]);
            if (null == fileName) {
                fileExistsChecks[i] = false;
                fileName = "no file";
            } else {
                fileExistsChecks[i] = true;
            }
            fileViews[i].setText(fileTextPrefix + fileName);
        }

        Bundle bundle = getIntent().getExtras();

        Button startButton = binding.startButton;
        startButton.setOnClickListener(startButtonListener);

        IPAddressView = binding.IPTextView;
        ArrayAdapter adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1
                , IPAddresses);
        IPAddressView.setAdapter(adapter);
        IPAddressView.setOnEditorActionListener(editorActionListener);

        if (null != bundle && bundle.containsKey("IP")) {
            IPAddressView.setText(bundle.getString("IP"));
        }

        Switch modeSwitch = binding.disableSwitch;
        modeSwitch.setOnClickListener(modeListener);

        if (null != bundle && bundle.containsKey(DISABLEREMOTEKEY)) {
            boolean state = bundle.getBoolean(DISABLEREMOTEKEY);
            Log.println(Log.INFO, "startupactivity", "setting disableRemote to: " + state);
            modeSwitch.setChecked(state);
            disableRemote = state;
        } else {
            disableRemote = false;
        }


        enableLoggingSwitch = binding.disableLoggingSwitch;
        enableLoggingSwitch.setOnClickListener(URIListener);

        enableLogging = (null != bundle && bundle.containsKey(ENABLELOGGINGKEY)
                && (bundle.getBoolean(ENABLELOGGINGKEY)));
        Log.println(Log.INFO, "startupactivity", "setting logging to: " + enableLogging);
        enableLoggingSwitch.setChecked(enableLogging);

        // button to select log files for the pocl image processor
        ActivityResultLauncher<Intent> resultLauncher;
        Button[] buttons = {binding.poclSelectURI, binding.monitorSelectURI,
                binding.cameraLogSelectURI};

        for(int i = 0; i< TOTALLOGS; i++) {
            resultLauncher =
                    registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                            new HandleActivityResultCallback(i));
            buttons[i].setOnClickListener(new SelectURIListener(filePrefixes[i], resultLauncher));
        }

        Button benchmarkButton = binding.BenchmarkButton;
        benchmarkButton.setOnClickListener(startBenchmarkListener);

        yuvCompButton = binding.yuvCompButton;
        yuvCompButton.setOnClickListener(yuvCompButtonListener);
        jpegCompButton = binding.jpegCompButton;
        jpegCompButton.setOnClickListener(jpegCompButtonListener);
        jpegImageButton = binding.jpegImageButton;
        jpegImageButton.setOnClickListener(jpegImageButtonListener);

        cameraLogSwitch = binding.cameraLogSwitch;

        // code to handle the camera JPEG quality input
        DropEditText qualityText = binding.jpegQualityEditText;
        qualityText.setOnEditorActionListener(jpegQualityTextListener);
        qualityText.setOnFocusChangeListener(jpegQualityFocusListener);

        jpegQuality = configStore.getJpegQuality();
        qualityText.setText(Integer.toString(jpegQuality));

    }

    private final View.OnClickListener jpegImageButtonListener = new View.OnClickListener() {

        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started jpegImageButtonListener callback");
            }

            if (jpegCompButton.isChecked()) {
                jpegCompButton.setChecked(false);
            }

            if (yuvCompButton.isChecked()) {
                yuvCompButton.setChecked(false);
            }
        }
    };

    /**
     * a callback when the yuvCompButton is pressed. It forces mutual exclusivity between it and the
     * jpecCompButton
     */
    private final View.OnClickListener yuvCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started yuvCompButtonListener callback");
            }

            if (jpegCompButton.isChecked()) {
                jpegCompButton.setChecked(false);
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }
        }
    };

    /**
     * a callback when the jpecCompButton is pressed. It forces mutual exclusivity between it and the
     * yuvCompButton
     */
    private final View.OnClickListener jpegCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started jpegCompButtonListener callback");
            }

            if (yuvCompButton.isChecked()) {
                yuvCompButton.setChecked(false);
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }
        }
    };

    /**
     * A callback that handles the camera JPEG quality edittext on screen when it loses focus.
     * This callback checks the input and sets it within the bounds of 1 - 100.
     */
    private final View.OnFocusChangeListener jpegQualityFocusListener =
            new View.OnFocusChangeListener() {
                @Override
                public void onFocusChange(View v, boolean hasFocus) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started qualityTextListener callback");
                    }

                    if (!hasFocus) {
                        TextView textView = (DropEditText) v;
                        int qualityInput;
                        try {
                            qualityInput = Integer.parseInt(textView.getText().toString());
                        } catch (Exception e) {
                            if (VERBOSITY >= 3) {
                                Log.println(Log.INFO, "StartupActivity.java", "could not parse quality, " +
                                        "defaulting to 80");
                            }
                            qualityInput = 80;
                            textView.setText(Integer.toString(qualityInput));
                        }

                        if (qualityInput < 1) {
                            qualityInput = 1;
                            textView.setText(Integer.toString(qualityInput));
                        } else if (qualityInput > 100) {
                            qualityInput = 100;
                            textView.setText(Integer.toString(qualityInput));
                        }

                        jpegQuality = qualityInput;
                    }
                }
            };

    /**
     * A callback that loses focus when the done button is pressed on a TextView.
     */
    private final TextView.OnEditorActionListener jpegQualityTextListener =
            new TextView.OnEditorActionListener() {
                @Override
                public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started jpegQualityTextListener " +
                                "callback");
                    }

                    if (EditorInfo.IME_ACTION_DONE == actionId) {
                        v.clearFocus();
                    }
                    return false;
                }
            };

    /**
     * get the name of the file pointed to by the uri.
     *
     * @param uri
     * @return filename or null
     */
    private String getFileName(Uri uri) {
        String fileName = null;
        if (null == uri) {
            return fileName;
        }

        // android mediastore API is basically a database,
        // therefore we need to go through all this trouble
        // the name of the file.
        Cursor cursor = getContentResolver()
                .query(uri, null, null, null, null, null);

        if (null == cursor) {
            return fileName;
        }

        try {
            if (cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);

                if (columnIndex >= 0) {
                    fileName = cursor.getString(columnIndex);
                }

            }
        } finally {
            cursor.close();

        }
        return fileName;

    }

    /**
     * A listener that on the press of a button; checks input variables
     * and starts the demo if they are valid.
     */
    private final View.OnClickListener startButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started startbuttonlisener callback");
            }

            String value = IPAddressView.getText().toString();

            boolean validInput = validateInput(value);
            if (!validInput) {
                return;
            }

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();
            Intent i = new Intent(getApplicationContext(), MainActivity.class);

            // pass ip to the main activity
            i.putExtra(IPKEY, value);
            i.putExtra(DISABLEREMOTEKEY, disableRemote);
            i.putExtra(ENABLELOGGINGKEY, enableLogging);

            int configFlag = genConfigFlags();

            if (enableLogging) {
                String uriString = uris[0].toString();
                i.putExtra(POCLLOGFILEURIKEY, uriString);
                uriString = uris[1].toString();
                i.putExtra(MONITORLOGFILEURIKEY, uriString);

                // only add it if camera logging is enabled
                if(cameraLogSwitch.isChecked()) {
                    uriString = uris[2].toString();
                    i.putExtra(CAMERALOGFILEURIKEY, uriString);
                }

            }

            configStore.setConfigFlags(configFlag);
            configStore.setJpegQuality(jpegQuality);
            // settings are only saved when calling this function.
            configStore.flushSetting();

            // start the main activity
            startActivity(i);
        }
    };

    private final View.OnClickListener startBenchmarkListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started startBenchmarkListener callback");
            }

            if (!enableLogging) {
                enableLogging = true;
                enableLoggingSwitch.setChecked(true);
            }

            String value = IPAddressView.getText().toString();

            boolean validInput = validateInput(value);
            if (!validInput) {
                return;
            }

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();
            Intent i = new Intent(getApplicationContext(), BenchmarkConfigurationActivity.class);

            // pass ip to the activity
            i.putExtra(IPKEY, value);
            i.putExtra(DISABLEREMOTEKEY, disableRemote);
            i.putExtra(ENABLELOGGINGKEY, enableLogging);
            String uriString = uris[0].toString();
            i.putExtra(POCLLOGFILEURIKEY, uriString);
            uriString = uris[1].toString();
            i.putExtra(MONITORLOGFILEURIKEY, uriString);

            int configFlags = genConfigFlags();
            configStore.setConfigFlags(configFlags);
            configStore.setJpegQuality(jpegQuality);
            // settings are only saved when calling this function.
            configStore.flushSetting();

            // start the main activity
            startActivity(i);
        }
    };

    /**
     * a function to generate configflags from the buttons
     * @return a valid configflag
     */
    int genConfigFlags() {
        int configFlag = NO_COMPRESSION;
        if (yuvCompButton.isChecked()) {
            configFlag |= YUV_COMPRESSION;
        }
        if (jpegCompButton.isChecked()) {
            configFlag |= JPEG_COMPRESSION;
        }
        if (jpegImageButton.isChecked()) {
            configFlag |= JPEG_IMAGE;
        }
        if (enableLogging) {
            configFlag |= ENABLE_PROFILING;
        }
        return configFlag;
    }

    /**
     * listener that causes the ip textview to lose focus once the
     * user presses done.
     */
    private final TextView.OnEditorActionListener editorActionListener =
            new TextView.OnEditorActionListener() {
        @Override
        public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started editorActionListener callback");
            }

            if (EditorInfo.IME_ACTION_DONE == actionId) {
                v.clearFocus();
            }
            return false;
        }
    };

    /**
     * A listener that hands interactions with the mode switch.
     * This switch sets the option to disable remote
     */
    private final View.OnClickListener modeListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started modelistener callback");
            }

            disableRemote = ((Switch) v).isChecked();
        }
    };

    /**
     * open a file create activity that lets
     * the user pick where to put the logging file.
     */
    private class SelectURIListener implements View.OnClickListener {

        private final String filePrefix;

        private final ActivityResultLauncher launcher;

        public SelectURIListener(String filePrefix, ActivityResultLauncher launcher) {
            this.filePrefix = filePrefix;
            this.launcher = launcher;
        }

        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started select" + filePrefix +
                        "URIListener callback");
            }

            // https://developer.android.com/training/data-storage/shared/documents-files
            Intent fileIntent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            fileIntent.addCategory(Intent.CATEGORY_OPENABLE);
            fileIntent.setType("text/comma-separated-values");
            fileIntent.putExtra(Intent.EXTRA_TITLE, filePrefix + "_log_file.csv");

            launcher.launch(fileIntent);
        }
    }

    /**
     * callback function that handles getting the uri from the document creation activity
     */
    private class HandleActivityResultCallback implements ActivityResultCallback<ActivityResult> {

        int id;

        /**
         * constructor of activity result
         *
         * @param id the index of the file to be used
         */
        public HandleActivityResultCallback(int id) {
            this.id = id;
        }

        @Override
        public void onActivityResult(ActivityResult result) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started " +
                        "handleActivityResultCallback callback");
            }

            if (Activity.RESULT_OK == result.getResultCode()) {

                Intent resultIntent = result.getData();

                if (null != resultIntent) {
                    uris[id] = resultIntent.getData();
                    // make permission to the file persist across phone reboots
                    getContentResolver().takePersistableUriPermission(uris[id],
                            Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                                    Intent.FLAG_GRANT_READ_URI_PERMISSION);

                    String fileName = getFileName(uris[id]);
                    fileViews[id].setText(fileTextPrefix + fileName);

                    fileExistsChecks[id] = true;

                    SharedPreferences.Editor editor = sharedPreferences.edit();
                    editor.putString(preferencekeys[id], uris[id].toString());
                    editor.apply();
                }
            }
        }
    }


    private boolean validateInput(String value) {
        // check that the input is ipv4
        boolean numeric = true;
        for (int i = 0; i < value.length(); i++) {
            char curChar = value.charAt(i);
            if (!isDigit(curChar) && curChar != '.') {
                numeric = false;
                break;
            }
        }

        if (!numeric) {
            Toast.makeText(StartupActivity.this, "Not a valid IP address, please check input",
                    Toast.LENGTH_SHORT).show();
            return false;
        }

        // check if all files exist
        boolean fileCheck = true;
        for (int i = 0; i < TOTALLOGS; i++) {
            fileCheck &= fileExistsChecks[i];
        }

        // handle cases where logging is enabled but file does not exist
        if (!fileCheck && enableLogging) {
            Toast.makeText(StartupActivity.this, "log file doesn't exist",
                    Toast.LENGTH_SHORT).show();
            return false;
        }

        return true;
    }

    /**
     * A listener that on the press of a button, will enable logging.
     */
    private final View.OnClickListener URIListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started urilistener callback");
            }

            enableLogging = ((Switch) v).isChecked();
        }
    };

    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started StartupActivity onDestroy method");
        }
        super.onDestroy();
    }
}
