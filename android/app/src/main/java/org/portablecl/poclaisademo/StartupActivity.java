package org.portablecl.poclaisademo;

import static java.lang.Character.isDigit;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.portablecl.poclaisademo.databinding.ActivityStartupBinding;


public class StartupActivity extends AppCompatActivity {

    /**
     * enable execution debug prints
     */
    private static final boolean DEBUGEXECUTION = true;

    /**
     * textview where the user inputs the ip address
     */
    private AutoCompleteTextView IPAddressView;

    /**
     * suggestion ip addresses
     */
    private final static String[] IPAddresses = {"192.168.36.206", "10.1.200.5"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityStartupBinding binding = ActivityStartupBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        Button startButton = binding.startButton;
        startButton.setOnClickListener(startButtonListener);

        IPAddressView = binding.IPTextView;
        ArrayAdapter adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, IPAddresses);
        IPAddressView.setAdapter(adapter);
        IPAddressView.setOnEditorActionListener(editorActionListener);

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

            Toast.makeText(StartupActivity.this, "Starting demo, please wait",
                    Toast.LENGTH_SHORT).show();
            Intent i = new Intent(getApplicationContext(), MainActivity.class);

            // pass ip to the main activity
            i.putExtra("IP", value);

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


    @Override
    protected void onDestroy() {

        if (DEBUGEXECUTION) {
            Log.println(Log.INFO, "EXECUTIONFLOW", "started StartupActivity onDestroy method");
        }
        super.onDestroy();
    }
}