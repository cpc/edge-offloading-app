package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.DISABLEREMOTEKEY;
import static org.portablecl.poclaisademo.BundleKeys.IPKEY;
import static org.portablecl.poclaisademo.BundleKeys.LOGFILEURIKEY;
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

import androidx.appcompat.app.AppCompatActivity;

import org.portablecl.poclaisademo.databinding.ActivityStartupBinding;


public class StartupActivity extends AppCompatActivity {

    private static final String URIPREFERENCEKEY = "org.portablecl.poclaisademo.logfile.urikey";

    /**
     * static prefix to use with displaying the file name
     */
    private static final String fileTextPrefix = "file name: ";

    /**
     * used in identifying the activity used to create a uri
     */
    private static final int CREATEDOCUMENTACTIVITYCODE = 1;

    /**
     * enable execution debug prints
     */
    private static final boolean DEBUGEXECUTION = true;

    /**
     * textview where the user inputs the ip address
     */
    private AutoCompleteTextView IPAddressView;

    /**
     * used to display the name of the file chosen
     */
    private TextView fileView;

    /**
     * suggestion ip addresses
     */
    private final static String[] IPAddresses = {"192.168.36.206", "10.1.200.5"};

    /**
     * a boolean that is passed to the main activity disable remote
     */
    private boolean disableRemote;

    /**
     * Universal Resource Identifier (URI) used to point the log file.
     * In newer android versions, you can not point to a file with paths.
     * Instead, you have to get/generate an URI which points to it.
     */
    private Uri uri = null;

    /**
     * boolean to set when a file is found
     */
    private boolean fileExists;

    /**
     * boolean to store user values
     */
    private boolean enableLogging;

    /**
     * value store that persists across app life cycles
     */
    private SharedPreferences sharedPreferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityStartupBinding binding = ActivityStartupBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // load uri from app preferences. This keeps the data between, app restarts.
        //https://developer.android.com/training/data-storage/shared-preferences
        sharedPreferences = getPreferences(Context.MODE_PRIVATE);
        try {
            uri = Uri.parse(sharedPreferences.getString(URIPREFERENCEKEY, null));

        } catch (Exception e) {
            Log.println(Log.INFO, "startupactivity", "could not parse stored uri");
            uri = null;
        }

        // display file name
        fileView = binding.fileNameView;
        String fileName = getFileName(uri);
        if (null == fileName) {
            fileExists = false;
            fileName = "no file";
        } else {
            fileExists = true;
        }
        fileView.setText(fileTextPrefix + fileName);

        Bundle bundle = getIntent().getExtras();

        Button startButton = binding.startButton;
        startButton.setOnClickListener(startButtonListener);

        IPAddressView = binding.IPTextView;
        ArrayAdapter adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1
                , IPAddresses);
        IPAddressView.setAdapter(adapter);
        IPAddressView.setOnEditorActionListener(editorActionListener);

        if (null != bundle && bundle.containsKey("IP")){
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


        Switch loggingSwitch = binding.disableLoggingSwitch;
        loggingSwitch.setOnClickListener(URIListener);

        enableLogging = (null != bundle && bundle.containsKey(LOGFILEURIKEY)
                && (null != bundle.getString(LOGFILEURIKEY)));
        Log.println(Log.INFO, "startupactivity", "setting disableRemote to: " + enableLogging);
        loggingSwitch.setChecked(enableLogging);

        Button selectURI = binding.selectURI;
        selectURI.setOnClickListener(selectURIListener);
    }

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
                return;
            }

            // handle cases where logging is enabled but file does not exist
            if (!fileExists && enableLogging) {
                Toast.makeText(StartupActivity.this, "log file doesn't exist",
                        Toast.LENGTH_SHORT).show();
                return;
            }

            // get string if logging is enabled
            String uriString = null;
            if (enableLogging) {
                uriString = uri.toString();
            }

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();
            Intent i = new Intent(getApplicationContext(), MainActivity.class);

            // pass ip to the main activity
            i.putExtra(IPKEY, value);
            i.putExtra(DISABLEREMOTEKEY, disableRemote);
            i.putExtra(LOGFILEURIKEY, uriString);

            // start the main activity
            startActivity(i);
        }
    };

    /**
     * listener that causes the ip textview to lose focus once the
     * user presses done.
     */
    private final TextView.OnEditorActionListener editorActionListener = new TextView.OnEditorActionListener() {
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
    private final View.OnClickListener selectURIListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started selectURIListener callback");
            }

            // https://developer.android.com/training/data-storage/shared/documents-files
            Intent fileIntent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            fileIntent.addCategory(Intent.CATEGORY_OPENABLE);
            fileIntent.setType("text/plain");
            fileIntent.putExtra(Intent.EXTRA_TITLE, "pocl_log_file.txt");
            // todo: use modern api call
            startActivityForResult(fileIntent, CREATEDOCUMENTACTIVITYCODE);

        }
    };

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
     * This method is overloaded to receive uri of the file the user has selected.
     *
     * @param requestCode The integer request code originally supplied to
     *                    startActivityForResult(), allowing you to identify who this
     *                    result came from.
     * @param resultCode  The integer result code returned by the child activity
     *                    through its setResult().
     * @param resultData  An Intent, which can return result data to the caller
     *                    (various data can be attached to Intent "extras").
     */
    @Override
    public void onActivityResult(int requestCode, int resultCode,
                                 Intent resultData) {
        super.onActivityResult(requestCode, resultCode, resultData);
        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started startbuttonlisener callback");
        }
        if (CREATEDOCUMENTACTIVITYCODE == requestCode && Activity.RESULT_OK == resultCode) {

            if (null != resultData) {
                uri = resultData.getData();
                // make permission to the file persist across phone reboots
                getContentResolver().takePersistableUriPermission(uri,
                        Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);

                String fileName = getFileName(uri);
                if (null == fileName) {
                    fileExists = false;
                    fileName = "no file";
                } else {
                    fileExists = true;
                }
                fileView.setText(fileTextPrefix + fileName);

                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString(URIPREFERENCEKEY, uri.toString());
                editor.apply();
            }
        }
    }

    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started StartupActivity onDestroy method");
        }
        super.onDestroy();
    }
}