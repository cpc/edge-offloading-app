package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.CAMERALOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLELOGGINGKEY;
import static org.portablecl.poclaisademo.BundleKeys.IPKEY;
import static org.portablecl.poclaisademo.BundleKeys.MONITORLOGFILEURIKEY;
import static org.portablecl.poclaisademo.BundleKeys.POCLLOGFILEURIKEY;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;
import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.ENABLE_PROFILING;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.JPEG_IMAGE;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.NO_COMPRESSION;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.YUV_COMPRESSION;
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
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
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
    private final static String[] IPAddresses = {"192.168.36.206", "192.168.50.112", "10.1.200.5"};

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
    /**
     * textview where the user inputs the ip address
     */
    private AutoCompleteTextView IPAddressView;
    /**
     * a boolean that is passed to the main activity disable remote
     */
    private boolean disableRemote;
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
    private ToggleButton jpegImageButton;
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
     * a callback when the jpecCompButton is pressed. It forces mutual exclusivity between it and
     * the
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

    private Button startButton;
    private Button benchmarkButton;
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
                if (!enableLogging) {
                    enableLogging = true;
                    enableLoggingSwitch.setChecked(true);
                }

                i = new Intent(getApplicationContext(), BenchmarkConfigurationActivity.class);
            } else if (v == startButton) {
                i = new Intent(getApplicationContext(), MainActivity.class);
            } else {
                Log.println(Log.ERROR, "startlistener", "unknown view type");
                return;
            }

            String value = IPAddressView.getText().toString();
            boolean validInput = validateInput(value);
            if (!validInput) {
                return;
            }

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();

            // pass ip to the main activity
            i.putExtra(IPKEY, value);
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
            // settings are only saved when calling this function.
            configStore.flushSetting();

            // start the main activity
            startActivity(i);
        }
    };

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

        enableLogging =
                (null != bundle && bundle.containsKey(ENABLELOGGINGKEY) && (bundle.getBoolean(ENABLELOGGINGKEY)));
        Log.println(Log.INFO, "startupactivity", "setting logging to: " + enableLogging);
        enableLoggingSwitch.setChecked(enableLogging);

        yuvCompButton = binding.yuvCompButton;
        yuvCompButton.setOnClickListener(yuvCompButtonListener);
        jpegCompButton = binding.jpegCompButton;
        jpegCompButton.setOnClickListener(jpegCompButtonListener);
        jpegImageButton = binding.jpegImageButton;
        jpegImageButton.setOnClickListener(jpegImageButtonListener);

        // code to handle the camera JPEG quality input
        DropEditText qualityText = binding.jpegQualityEditText;
        qualityText.setOnEditorActionListener(jpegQualityTextListener);
        qualityText.setOnFocusChangeListener(jpegQualityFocusListener);

        jpegQuality = configStore.getJpegQuality();
        qualityText.setText(Integer.toString(jpegQuality));

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
        return configFlag;
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

        return true;
    }

    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started StartupActivity onDestroy method");
        }
        super.onDestroy();
    }
}
