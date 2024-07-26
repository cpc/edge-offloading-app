package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.CAMERALOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLELOGGINGKEY;
import static org.portablecl.poclaisademo.BundleKeys.MONITORLOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.POCLLOGFILEURIKEY;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.ENABLE_PROFILING;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.LOCAL_ONLY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.SOFTWARE_HEVC_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;
import static org.portablecl.poclaisademo.PoclImageProcessor.sanitizePipelineLanes;
import static org.portablecl.poclaisademo.PoclImageProcessor.sanitizeTargetFPS;
import static java.lang.Character.isDigit;

import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.appcompat.app.AppCompatActivity;

import org.portablecl.poclaisademo.databinding.ActivityStartupBinding;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;


public class StartupActivity extends AppCompatActivity {
    /**
     * suggestion ip addresses
     */
    private final static String[] IPAddresses = {"192.168.36.206", "192.168.50.112", "10.1.200.5",
            "192.168.88.232", "192.168.88.217"};

    /**
     * A callback that loses focus when the done button is pressed on a TextView.
     */
    private final TextView.OnEditorActionListener loseFocusListener =
            new TextView.OnEditorActionListener() {
                @Override
                public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started loseFocusListener " +
                                "callback");
                    }

                    if (EditorInfo.IME_ACTION_DONE == actionId) {
                        v.clearFocus();
                    }
                    return false;
                }
            };
    /**
     * listener that causes the ip textview to lose focus once the
     * user presses done.
     */
    private final TextView.OnEditorActionListener editorActionListener =
            new TextView.OnEditorActionListener() {
                @Override
                public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {

                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started editorActionListener " +
                                "callback");
                    }

                    if (EditorInfo.IME_ACTION_DONE == actionId) {
                        v.clearFocus();
                    }
                    return false;
                }
            };
    private final View.OnClickListener runtimeEvalSwitchListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

        }
    };
    private final View.OnClickListener lockCodecSwitchListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

        }
    };
    DiscoverySelect DSSelect;
    /**
     * textview where the user inputs the ip address
     */
    private AutoCompleteTextView IPAddressView;
    /**
     * a boolean that is passed to the main activity disable remote
     */
    private boolean disableRemote;
    /**
     * boolean to store user values
     */
    private boolean enableLogging;
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
    /**
     * Quality parameter of camera's JPEG compression
     */
    private int jpegQuality;
    /**
     * A callback that handles the camera JPEG quality edittext on screen when it loses focus.
     * This callback checks the input and sets it within the bounds of 1 - 100.
     */
    private final View.OnFocusChangeListener jpegQualityFocusListener =
            new View.OnFocusChangeListener() {
                @Override
                public void onFocusChange(View v, boolean hasFocus) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started qualityTextListener " +
                                "callback");
                    }

                    if (!hasFocus) {
                        TextView textView = (DropEditText) v;
                        int qualityInput;
                        try {
                            qualityInput = Integer.parseInt(textView.getText().toString());
                        } catch (Exception e) {
                            if (VERBOSITY >= 3) {
                                Log.println(Log.INFO, "StartupActivity.java", "could not parse " +
                                        "quality, " + "defaulting to 80");
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
    private Switch enableLoggingSwitch;
    private ConfigStore configStore;
    private ToggleButton yuvCompButton;
    private ToggleButton jpegCompButton;

    private ToggleButton hevcCompButton;

    private ToggleButton softwareHevcCompButton;
    /**
     * callback function that disables compression options not compatible with the jpeg image
     */
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

            if (hevcCompButton.isChecked()) {
                hevcCompButton.setChecked(false);
            }

            if (softwareHevcCompButton.isChecked()) {
                softwareHevcCompButton.setChecked(false);
            }
        }
    };
    private int targetFPS;
    /**
     * A callback that handles the targetFPS edittext on screen when it loses focus.
     * This callback checks the input and sets it within the bounds.
     */
    private final View.OnFocusChangeListener targetFPSFocusListener =
            new View.OnFocusChangeListener() {
                @Override
                public void onFocusChange(View v, boolean hasFocus) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started targetFPSFocusListener " +
                                "callback");
                    }

                    if (!hasFocus) {
                        TextView textView = (DropEditText) v;
                        int value = Integer.MAX_VALUE;
                        try {
                            value = Integer.parseInt(textView.getText().toString());
                        } catch (Exception e) {
                            if (VERBOSITY >= 3) {
                                Log.println(Log.INFO, "StartupActivity.java", "could not parse " +
                                        "target fps, defaulting");
                            }

                        }

                        // make sure that the value is not an unreasonable value
                        int sanitizedValue = sanitizeTargetFPS(value);
                        if (sanitizedValue != value) {
                            textView.setText(Integer.toString(sanitizedValue));
                        }

                        targetFPS = sanitizedValue;
                    }
                }
            };
    private int pipelineLanes;
    /**
     * A callback that handles the pipeline edittext on screen when it loses focus.
     * This callback checks the input and sets it within the bounds.
     */
    private final View.OnFocusChangeListener pipelineFocusListener =
            new View.OnFocusChangeListener() {
                @Override
                public void onFocusChange(View v, boolean hasFocus) {
                    if (DEBUGEXECUTION) {
                        Log.println(Log.INFO, "EXECUTIONFLOW", "started pipelineFocusListener " +
                                "callback");
                    }

                    if (!hasFocus) {
                        TextView textView = (DropEditText) v;
                        int value = Integer.MIN_VALUE;
                        try {
                            value = Integer.parseInt(textView.getText().toString());
                        } catch (Exception e) {
                            if (VERBOSITY >= 3) {
                                Log.println(Log.INFO, "StartupActivity.java", "could not parse " +
                                        "number of lanes, defaulting");
                            }

                        }

                        // make sure that the value is not an unreasonable value
                        int sanitizedValue = sanitizePipelineLanes(value);
                        if (sanitizedValue != value) {
                            textView.setText(Integer.toString(sanitizedValue));
                        }

                        pipelineLanes = sanitizedValue;
                    }
                }
            };
    private ToggleButton jpegImageButton;
    /**
     * a callback when the yuvCompButton is pressed. It disables buttons not compatible with it
     */
    private final View.OnClickListener yuvCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started yuvCompButtonListener callback");
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }

        }
    };
    /**
     * a callback when the yuvCompButton is pressed. It disables buttons not compatible with it
     */
    private final View.OnClickListener hevcCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started yuvCompButtonListener callback");
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }

            // disable for now since they share the same decoder
            // and currently, it only supports one instance
            if (softwareHevcCompButton.isChecked()) {
                softwareHevcCompButton.setChecked(false);
            }

        }
    };
    /**
     * a callback when the yuvCompButton is pressed. It disables buttons not compatible with it
     */
    private final View.OnClickListener softwareHevcCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started yuvCompButtonListener callback");
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }

            // disable for now since they share the same decoder
            // and currently, it only supports one instance
            if (hevcCompButton.isChecked()) {
                hevcCompButton.setChecked(false);
            }

        }
    };
    /**
     * a callback when the jpecCompButton is pressed. It disables buttons not compatible with it
     */
    private final View.OnClickListener jpegCompButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started jpegCompButtonListener callback");
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }

        }
    };
    private Button startButton;
    private Button benchmarkButton;
    private Switch qualityAlgorithmSwitch;
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

            // these two switches are mutually exclusive
            if (qualityAlgorithmSwitch.isChecked() && ((Switch) v).isChecked()) {
                qualityAlgorithmSwitch.performClick();
            }

            enableAllCompButtons(!((Switch) v).isChecked());
            if (((Switch) v).isChecked()) {
                setCheckedAllCompButtons(false);
            }

            disableRemote = ((Switch) v).isChecked();

        }
    };
    private Switch modeSwitch;
    /**
     * a callback function that also configures the compression types
     */
    private final View.OnClickListener qualityAlgorithmSwitchListener = new View.OnClickListener() {

        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started qualityAlgorithmSwitchListener " +
                        "callback");
            }

            // local only and quality algorithm are mutually exclusive
            if (modeSwitch.isChecked() && ((Switch) v).isChecked()) {
                modeSwitch.performClick();
            }

            // disable or enable all comp buttons
            // if the quality algorithm button is turned on
            enableAllCompButtons(!((Switch) v).isChecked());

            if (!jpegCompButton.isChecked()) {
                jpegCompButton.setChecked(true);
            }

            if (!hevcCompButton.isChecked()) {
                hevcCompButton.setChecked(true);
            }

            if (softwareHevcCompButton.isChecked()) {
                softwareHevcCompButton.setChecked(false);
            }

            if (yuvCompButton.isChecked()) {
                yuvCompButton.setChecked(false);
            }

            if (jpegImageButton.isChecked()) {
                jpegImageButton.setChecked(false);
            }
        }
    };
    private Switch runtimeEvalSwitch;
    private Switch lockCodecSwitch;
    /**
     * A listener that on the press of a button; checks input variables
     * and starts the demo if they are valid.
     */
    private final View.OnClickListener startListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started startlisener callback");
            }

            Intent i;
            if (v == benchmarkButton) {
                i = new Intent(getApplicationContext(), BenchmarkConfigurationActivity.class);
            } else if (v == startButton) {
                i = new Intent(getApplicationContext(), MainActivity.class);
                // wipe the calibration uri
                configStore.setCalibrateVideoUri(null);
            } else {
                Log.println(Log.ERROR, "startlistener", "unknown view type");
                return;
            }

            String value = IPAddressView.getText().toString();
            boolean validInput = validateInput(value);
            if (!validInput) {
                return;
            }

            if (!value.contains(":") && value.length() != 0) {
                value = value + ":10998";
            }

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();

            // pass ip to the main activity
            // todo: store these values in the configstore instead
            i.putExtra(DISABLEREMOTEKEY, disableRemote);
            i.putExtra(ENABLELOGGINGKEY, enableLogging);

            int configFlag = genConfigFlags();

            LocalDateTime datetime = LocalDateTime.now();
            String datetimeText = datetime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T" +
                    "'HH_mm_ss"));

            if (enableLogging) {
                String pocl_file = "pocl_log_" + datetimeText + ".csv";
                Uri uri = createLogFile(pocl_file);
                i.putExtra(POCLLOGFILEURIKEY, uri.toString());

                String monitor_file = "monitor_log_" + datetimeText + ".csv";
                uri = createLogFile(monitor_file);
                i.putExtra(MONITORLOGFILEURIKEY, uri.toString());

                String camera_file = "camera_log_" + datetimeText + ".csv";
                uri = createLogFile(camera_file);
                i.putExtra(CAMERALOGFILEURIKEY, uri.toString());

            }

            configStore.setConfigFlags(configFlag);
            configStore.setJpegQuality(jpegQuality);
            configStore.setIpAddressText(value);
            configStore.setQualityAlgorithmOption(qualityAlgorithmSwitch.isChecked());
            configStore.setRuntimeEvalOption(runtimeEvalSwitch.isChecked());
            configStore.setLockCodecOption(lockCodecSwitch.isChecked());
            configStore.setTargetFPS(targetFPS);
            configStore.setPipelineLanes(pipelineLanes);
            // settings are only saved when calling this function.
            configStore.flushSetting();

            // start the main activity
            startActivity(i);
        }
    };

    /**
     * setClickable on every compButton
     *
     * @param value set false to disable clicking
     */
    private void enableAllCompButtons(boolean value) {
        jpegCompButton.setClickable(value);
        jpegImageButton.setClickable(value);
        hevcCompButton.setClickable(value);
        yuvCompButton.setClickable(value);
        softwareHevcCompButton.setClickable(value);
    }

    /**
     * set checked on all buttons
     *
     * @param value true or false
     */
    private void setCheckedAllCompButtons(boolean value) {
        jpegCompButton.setChecked(value);
        jpegImageButton.setChecked(value);
        hevcCompButton.setChecked(value);
        softwareHevcCompButton.setChecked(value);
        yuvCompButton.setChecked(value);

    }

    /**
     * function to create a csv log file
     *
     * @param fileName name of the file
     * @return an uri to the log file
     */
    private Uri createLogFile(String fileName) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName); // this is the name
        // of the file
        values.put(MediaStore.MediaColumns.MIME_TYPE, "text/comma-separated-values");
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
        Uri uri = getContentResolver().insert(MediaStore.Files.getContentUri("external"),
                values);

        return uri;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityStartupBinding binding = ActivityStartupBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        configStore = new ConfigStore(this);

        Bundle bundle = getIntent().getExtras();

        startButton = binding.startButton;
        startButton.setOnClickListener(startListener);
        benchmarkButton = binding.BenchmarkButton;
        benchmarkButton.setOnClickListener(startListener);

        IPAddressView = binding.IPTextView;
        ArrayAdapter adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1
                , IPAddresses);
        IPAddressView.setAdapter(adapter);
        IPAddressView.setOnEditorActionListener(editorActionListener);
        IPAddressView.setText(configStore.getIpAddressText());

        modeSwitch = binding.disableSwitch;
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

        enableLogging =
                (null != bundle && bundle.containsKey(ENABLELOGGINGKEY) && (bundle.getBoolean(ENABLELOGGINGKEY)));
        Log.println(Log.INFO, "startupactivity", "setting logging to: " + enableLogging);
        enableLoggingSwitch.setChecked(enableLogging);

        int configFlags = configStore.getConfigFlags();

        yuvCompButton = binding.yuvCompButton;
        yuvCompButton.setOnClickListener(yuvCompButtonListener);
        yuvCompButton.setChecked((YUV_COMPRESSION & configFlags) > 0);

        jpegCompButton = binding.jpegCompButton;
        jpegCompButton.setOnClickListener(jpegCompButtonListener);
        jpegCompButton.setChecked((JPEG_COMPRESSION & configFlags) > 0);

        jpegImageButton = binding.jpegImageButton;
        jpegImageButton.setOnClickListener(jpegImageButtonListener);
        jpegImageButton.setChecked((JPEG_IMAGE & configFlags) > 0);
        // disable jpeg image compression
        jpegImageButton.setClickable(false);

        hevcCompButton = binding.hevcCompButton;
        hevcCompButton.setOnClickListener(hevcCompButtonListener);
        hevcCompButton.setChecked((HEVC_COMPRESSION & configFlags) > 0);

        softwareHevcCompButton = binding.softwareHevcCompButton;
        softwareHevcCompButton.setOnClickListener(softwareHevcCompButtonListener);
        softwareHevcCompButton.setChecked((SOFTWARE_HEVC_COMPRESSION & configFlags) > 0);

        // code to handle the camera JPEG quality input
        DropEditText qualityText = binding.jpegQualityEditText;
        qualityText.setOnEditorActionListener(loseFocusListener);
        qualityText.setOnFocusChangeListener(jpegQualityFocusListener);
        jpegQuality = configStore.getJpegQuality();
        qualityText.setText(Integer.toString(jpegQuality));

        qualityAlgorithmSwitch = binding.qualityAlgorithmSwitch;
        qualityAlgorithmSwitch.setOnClickListener(qualityAlgorithmSwitchListener);
        if (configStore.getQualityAlgorithmOption()) {
            qualityAlgorithmSwitch.performClick();
        }

        runtimeEvalSwitch = binding.runtimeEvalSwitch;
        runtimeEvalSwitch.setOnClickListener(runtimeEvalSwitchListener);
        if (configStore.getRuntimeEvalOption()) {
            runtimeEvalSwitch.performClick();
        }

        lockCodecSwitch = binding.lockCodecSwitch;
        lockCodecSwitch.setOnClickListener(lockCodecSwitchListener);
        if (configStore.getLockCodecOption()) {
            lockCodecSwitch.performClick();
        }


        targetFPS = configStore.getTargetFPS();
        DropEditText targetFPSField = binding.targetFPS;
        targetFPSField.setText(Integer.toString(targetFPS));
        targetFPSField.setOnEditorActionListener(loseFocusListener);
        targetFPSField.setOnFocusChangeListener(targetFPSFocusListener);

        pipelineLanes = configStore.getPipelineLanes();
        DropEditText pipelineLanesField = binding.pipelineLane;
        pipelineLanesField.setText(Integer.toString(pipelineLanes));
        pipelineLanesField.setOnEditorActionListener(loseFocusListener);
        pipelineLanesField.setOnFocusChangeListener(pipelineFocusListener);

        Spinner discoverySpinner = binding.discoverySpinner;
        // Create a listener callback for the spinner object to use the selected server from the
        // list.
        //  The callback is passed to an instance of DiscoverSelect class: DSSelect.
        DSSelect = new DiscoverySelect(this, discoverySpinner,
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> parent, View view, int position,
                                               long id) {

                        String selectedServer = DSSelect.spinnerList.get(position).getAddress();
                        if (!selectedServer.equals(DiscoverySelect.DEFAULT_SPINNER_VAL)) {
                            Log.d("DISC", "Spinner position selected: " + position + " : server " +
                                    "selected " +
                                    ": " + selectedServer);
                            IPAddressView.setText(DSSelect.spinnerList.get(position).getAddress());
                        }
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {
                    }
                });
    }

    /**
     * a function to generate configflags from the buttons
     *
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
        if (hevcCompButton.isChecked()) {
            configFlag |= HEVC_COMPRESSION;
        }
        if (softwareHevcCompButton.isChecked()) {
            configFlag |= SOFTWARE_HEVC_COMPRESSION;
        }
        if (disableRemote) {
            configFlag |= LOCAL_ONLY;
        }

        return configFlag;
    }

    private boolean validateInput(String value) {
        // check that the input is ipv4. ip with and without port is accepted
        boolean numeric = true;
        for (int i = 0; i < value.length(); i++) {
            char curChar = value.charAt(i);
            if (!isDigit(curChar) && curChar != '.' && curChar != ':') {
                numeric = false;
                break;
            }
        }

        if (!numeric) {
            Toast.makeText(StartupActivity.this, "Not a valid IP address, please check input",
                    Toast.LENGTH_SHORT).show();
            return false;
        }

        return true;
    }

    @Override
    protected void onPause() {
        super.onPause();
        DSSelect.stopDiscovery();
    }

    @Override
    protected void onDestroy() {
        DSSelect.stopDiscovery();
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started StartupActivity onDestroy method");
        }
        super.onDestroy();
    }
}
