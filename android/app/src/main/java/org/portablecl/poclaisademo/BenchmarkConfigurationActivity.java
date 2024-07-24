package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.BundleKeys.BENCHMARKVIDEOURI;
import static org.portablecl.poclaisademo.BundleKeys.ENABLECOMPRESSIONKEY;
import static org.portablecl.poclaisademo.BundleKeys.ENABLESEGMENTATIONKEY;
import static org.portablecl.poclaisademo.BundleKeys.IMAGECAPTUREFRAMETIMEKEY;
import static org.portablecl.poclaisademo.DevelopmentVariables.DEBUGEXECUTION;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
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

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.portablecl.poclaisademo.databinding.ActivityBenchmarkConfigurationBinding;

public class BenchmarkConfigurationActivity extends AppCompatActivity {

    /**
     * static prefix to use with displaying the file name
     */
    private static final String fileTextPrefix = "file name: ";

    public static final int TOTALBENCHMARKS = 1;

    private final Uri[] benchmarkUris = new Uri[TOTALBENCHMARKS];

    private final TextView[] benchmarkFileViews = new TextView[TOTALBENCHMARKS];

    private AutoCompleteTextView framerateTextView;

    private final boolean[] benchmarkFileExistsChecks = new boolean[TOTALBENCHMARKS];

    private boolean enableCompression;

    private boolean enableSegmentation;

    /**
     * value store that persists across app life cycles
     */
    private SharedPreferences sharedPreferences;

    private static final String[] preferenceKeys = {
            "org.portablecl.poclaisademo.benchmark.videobenchmark",
            "org.portablecl.poclaisademo.benchmark.enable.compression",
            "org.portablecl.poclaisademo.benchmark.enablesegmentation"
    };

    private Bundle bundle;

    /**
     * default options for the framerate
     */
    private final static String[] frameRates = {"0", "100", "200", "300", "400", "500", "600",
            "700", "800", "900", "1000", "1100", "1200", "2000"};

    private Button mainActivityButton;
    private ConfigStore configStore;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        bundle = getIntent().getExtras();

        ActivityBenchmarkConfigurationBinding binding =
                ActivityBenchmarkConfigurationBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        configStore = new ConfigStore(this);

        sharedPreferences = getPreferences(Context.MODE_PRIVATE);

        // setup compression switch
        enableCompression = sharedPreferences.getBoolean(preferenceKeys[1], false);
        Switch compressionSwitch = binding.compressionSwitch;
        compressionSwitch.setChecked(enableCompression);
        compressionSwitch.setOnClickListener(compressionListener);

        // setup segmentation switch
        enableSegmentation = sharedPreferences.getBoolean(preferenceKeys[2], false);
        Switch segmentationSwitch = binding.segmentationSwitch;
        segmentationSwitch.setChecked(enableSegmentation);
        segmentationSwitch.setOnClickListener(segmentationListener);


        //  setup benchmark select button
        try {
            benchmarkUris[0] = Uri.parse(sharedPreferences.getString(preferenceKeys[0], null));
        } catch (Exception e) {
            Log.println(Log.INFO, "startupactivity",
                    "could not parse stored" + BENCHMARKVIDEOURI + " uri");
            benchmarkUris[0] = null;
        }

        benchmarkFileViews[0] = binding.videoSelectView;
        String fileName = getFileName(benchmarkUris[0]);
        if (null == fileName) {
            benchmarkFileExistsChecks[0] = false;
            fileName = "no file";
        } else {
            benchmarkFileExistsChecks[0] = true;
        }
        benchmarkFileViews[0].setText(fileTextPrefix + fileName);

        Button videoSelectButton = binding.videoSelectButton;
        ActivityResultLauncher launcher =
                registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                        new SelectVideoResultCallback(0));
        videoSelectButton.setOnClickListener(new VideoSelectListener(launcher));

        // setup start benchmark button
        Button startBenchmarkButton = binding.startBenchmarkButton;
        startBenchmarkButton.setOnClickListener(startBenchmarkButtonListener);

        framerateTextView = binding.framerateTextView;
        ArrayAdapter adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1,
                frameRates);
        framerateTextView.setAdapter(adapter);
        framerateTextView.setOnEditorActionListener(editorActionListener);

        mainActivityButton = binding.mainActivityButton;
        mainActivityButton.setOnClickListener(startMainActivityListener);
    }

    private final View.OnClickListener compressionListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            enableCompression = ((Switch) v).isChecked();
        }
    };

    private final View.OnClickListener segmentationListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            enableSegmentation = ((Switch) v).isChecked();
        }
    };

    /**
     * start the activity to pick the video to use as a benchmark
     */
    private class VideoSelectListener implements View.OnClickListener {

        private final ActivityResultLauncher launcher;

        public VideoSelectListener(ActivityResultLauncher launcher) {
            this.launcher = launcher;
        }

        @Override
        public void onClick(View v) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started select video callback");
            }

            // https://developer.android.com/training/data-storage/shared/documents-files
            Intent fileIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            fileIntent.addCategory(Intent.CATEGORY_OPENABLE);
//            fileIntent.setType("video/mp4");
            fileIntent.setType("application/octet-stream");

            launcher.launch(fileIntent);
        }
    }

    /**
     * callback function that handles getting the uri from the document creation activity
     */
    private class SelectVideoResultCallback implements ActivityResultCallback<ActivityResult> {

        int id;

        /**
         * constructor of activity result
         *
         * @param id the index of the file to be used
         */
        public SelectVideoResultCallback(int id) {
            this.id = id;
        }

        @Override
        public void onActivityResult(ActivityResult result) {
            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started " +
                        "selectVideoResultCallback callback");
            }

            if (Activity.RESULT_OK == result.getResultCode()) {

                Intent resultIntent = result.getData();

                if (null != resultIntent) {
                    benchmarkUris[id] = resultIntent.getData();
                    // make permission to the file persist across phone reboots
                    getContentResolver().takePersistableUriPermission(benchmarkUris[id],
                            Intent.FLAG_GRANT_READ_URI_PERMISSION);

                    String fileName = getFileName(benchmarkUris[id]);
                    benchmarkFileViews[id].setText(fileTextPrefix + fileName);

                    benchmarkFileExistsChecks[id] = true;
                }
            }
        }
    }

    private final View.OnClickListener startMainActivityListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started startBenchmarkButtonListener " +
                        "callback");
            }

            // check if all files exist
            boolean fileCheck = true;
            for (int i = 0; i < TOTALBENCHMARKS; i++) {
                fileCheck &= benchmarkFileExistsChecks[i];
            }

            if (!fileCheck) {
                Toast.makeText(BenchmarkConfigurationActivity.this, "benchmark file doesn't exist",
                        Toast.LENGTH_SHORT).show();
                return;
            }

            configStore.setCalibrateVideoUri(benchmarkUris[0].toString());
            configStore.flushSetting();

            Intent i = new Intent(getApplicationContext(), MainActivity.class);
            // pass everything passed to the benchmark configuration activity to the mainactivity
            i.putExtras(bundle);
            startActivity(i);
        }
    };

    /**
     * A listener that on the press of a button; checks input variables
     * and starts the demo if they are valid.
     */
    private final View.OnClickListener startBenchmarkButtonListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            if (DEBUGEXECUTION) {
                Log.println(Log.INFO, "EXECUTIONFLOW", "started startBenchmarkButtonListener " +
                        "callback");
            }

            int frameRate;
            try {
                frameRate = Integer.parseInt(framerateTextView.getText().toString());
            } catch (NumberFormatException e) {
                Toast.makeText(BenchmarkConfigurationActivity.this, "framerate could not be parsed",
                        Toast.LENGTH_SHORT).show();
                return;
            }

            // check if all files exist
            boolean fileCheck = true;
            for (int i = 0; i < TOTALBENCHMARKS; i++) {
                fileCheck &= benchmarkFileExistsChecks[i];
            }

            if (!fileCheck) {
                Toast.makeText(BenchmarkConfigurationActivity.this, "benchmark file doesn't exist",
                        Toast.LENGTH_SHORT).show();
                return;
            }

            // save preferences before doing anything else
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(preferenceKeys[0], benchmarkUris[0].toString());
            editor.putBoolean(preferenceKeys[1], enableCompression);
            editor.putBoolean(preferenceKeys[2], enableSegmentation);
            editor.apply();

            Toast.makeText(BenchmarkConfigurationActivity.this, "Starting benchmark, please wait",
                    Toast.LENGTH_SHORT).show();

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Intent serviceIntent = new Intent(getApplicationContext(), BenchmarkService.class);

                serviceIntent.putExtras(bundle);
                serviceIntent.putExtra(BENCHMARKVIDEOURI, benchmarkUris[0].toString());
                serviceIntent.putExtra(ENABLECOMPRESSIONKEY, enableCompression);
                serviceIntent.putExtra(ENABLESEGMENTATIONKEY, enableSegmentation);
                serviceIntent.putExtra(IMAGECAPTUREFRAMETIMEKEY, frameRate);

                // when everything works, set to foregroundservice
                getApplicationContext().startForegroundService(serviceIntent);
            } else {
                Toast.makeText(BenchmarkConfigurationActivity.this, "not doing benchmark, android" +
                                " version too low",
                        Toast.LENGTH_SHORT).show();
            }

        }
    };

    /**
     * lose focus once you press done on the keyboard
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
}