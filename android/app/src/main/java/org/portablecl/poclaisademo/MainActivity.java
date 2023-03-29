package org.portablecl.poclaisademo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import org.portablecl.poclaisademo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // These native functions are defined in src/main/cpp/vectorAddExample.cpp
    public native int initCL();
    public native int vectorAddCL(int N, float[] A, float[] B, float[] C);
    public native int destroyCL();
    public native void setenv(String key, String value);


    // Used to load the 'poclaisademo' library on application startup.
    static {
        System.loadLibrary("poclaisademo");
    }

    TextView text;

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());

        text = binding.clOutput;
        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };

        td.start();

    }

    /**
     * A native method that is implemented by the 'poclaisademo' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    void doVectorAdd()
    {
        // Error checkings are not done for simplicity. Check logcat

        printLog("\ncalling opencl init functions... ");
        initCL();

        // Create 2 vectors A & B
        // And yes, this array size is embarrassingly huge for demo!
        float A[] = {1, 2, 3, 4, 5, 6, 7};
        float B[] = {8, 9, 0, 6, 7, 8, 9};
        float C[] = new float[A.length];

        printLog("\n A: ");
        for(int i=0; i<A.length; i++)
            printLog(Float.toString(A[i]) + "    ");

        printLog("\n B: ");
        for(int i=0; i<B.length; i++)
            printLog(Float.toString(B[i]) + "    ");

        printLog("\n\ncalling opencl vector-addition kernel... ");
        vectorAddCL(C.length, A, B, C);

        printLog("\n C: ");
        for(int i=0; i<C.length; i++)
            printLog(Float.toString(C[i]) + "    ");

        boolean correct = true;
        for(int i=0; i<C.length; i++)
        {
            if(C[i] != (A[i] + B[i])) {
                correct = false;
                break;
            }
        }

        if(correct)
            printLog("\n\nresult: passed\n");
        else
            printLog("\n\nresult: failed\n");

        printLog("\ndestroy opencl resources... ");
        destroyCL();
    }


    void printLog(final String str)
    {
        // UI updates should happen only in UI thread
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                text.append(str);
            }
        });
    }
}