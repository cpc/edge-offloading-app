package org.portablecl.poclaisademo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;
import android.content.Context;

import org.portablecl.poclaisademo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // These native functions are defined in src/main/cpp/vectorAddExample.cpp
    public native int initCL();
    public native int vectorAddCL(int N, float[] A, float[] B, float[] C);
    public native int destroyCL();
    public native void setenv(String key, String value);

    public native int initPoCL();

    public native void setPoCLEnv(String key, String value);
    public native int poclRemoteVectorAdd(int N, float[] A, float[] B, float[] C);
    public native int destroyPoCL();


    // Used to load the 'poclaisademo' library on application startup.
    static {
        System.loadLibrary("poclaisademo");
        System.loadLibrary("poclremoteexample");
    }

    TextView ocl_text;
    TextView pocl_text;

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());

        ocl_text = binding.clOutput;
        pocl_text = binding.poclOutput;

        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };
        td.start();

        // todo: have this be a parameter set by the user
        String server_address = "192.168.50.112";

        String cache_dir = getCacheDir().getAbsolutePath();

        //configure environment variables pocl needs to run
        setPoCLEnv("POCL_DEBUG", "all");
        setPoCLEnv("POCL_DEVICES", "remote");
        setPoCLEnv("POCL_REMOTE0_PARAMETERS", server_address);
        setPoCLEnv("POCL_CACHE_DIR",cache_dir);

        Thread pocl_td =new Thread() {
            public void run() {
                doPoclVectorAdd();
            }
        };
        pocl_td.start();

    }

    /**
     * A native method that is implemented by the 'poclaisademo' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    void doVectorAdd()
    {
        // Error checkings are not done for simplicity. Check logcat

        printLog(ocl_text, "\ncalling opencl init functions... ");
        initCL();

        // Create 2 vectors A & B
        // And yes, this array size is embarrassingly huge for demo!
        float A[] = {1, 2, 3, 4, 5, 6, 7};
        float B[] = {8, 9, 0, 6, 7, 8, 9};
        float C[] = new float[A.length];

        printLog(ocl_text, "\n A: ");
        for(int i=0; i<A.length; i++)
            printLog(ocl_text, Float.toString(A[i]) + "    ");

        printLog(ocl_text, "\n B: ");
        for(int i=0; i<B.length; i++)
            printLog(ocl_text, Float.toString(B[i]) + "    ");

        printLog(ocl_text, "\n\ncalling opencl vector-addition kernel... ");
        vectorAddCL(C.length, A, B, C);

        printLog(ocl_text, "\n C: ");
        for(int i=0; i<C.length; i++)
            printLog(ocl_text, Float.toString(C[i]) + "    ");

        boolean correct = true;
        for(int i=0; i<C.length; i++)
        {
            if(C[i] != (A[i] + B[i])) {
                correct = false;
                break;
            }
        }

        if(correct)
            printLog(ocl_text, "\n\nresult: passed\n");
        else
            printLog(ocl_text, "\n\nresult: failed\n");

        printLog(ocl_text, "\ndestroy opencl resources... ");
        destroyCL();
    }


    void printLog(TextView tv, final String str)
    {
        // UI updates should happen only in UI thread
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tv.append(str);
            }
        });
    }


    void doPoclVectorAdd()
    {
        // Error checkings are not done for simplicity. Check logcat

        printLog(pocl_text, "\ncalling PoCL init functions... ");
        initPoCL();

        // Create 2 vectors A & B
        // And yes, this array size is embarrassingly huge for demo!
        float A[] = {1, 2, 3, 4, 5, 6, 7};
        float B[] = {8, 9, 0, 6, 7, 8, 9};
        float C[] = new float[A.length];

        printLog(pocl_text, "\n A: ");
        for(int i=0; i<A.length; i++)
            printLog(pocl_text, Float.toString(A[i]) + "    ");

        printLog(pocl_text, "\n B: ");
        for(int i=0; i<B.length; i++)
            printLog(pocl_text, Float.toString(B[i]) + "    ");

        printLog(pocl_text, "\n\ncalling opencl vector-addition kernel... ");
        poclRemoteVectorAdd(C.length, A, B, C);

        printLog(pocl_text, "\n C: ");
        for(int i=0; i<C.length; i++)
            printLog(pocl_text, Float.toString(C[i]) + "    ");

        boolean correct = true;
        for(int i=0; i<C.length; i++)
        {
            if(C[i] != (A[i] + B[i])) {
                correct = false;
                break;
            }
        }

        if(correct)
            printLog(pocl_text, "\n\nresult: passed\n");
        else
            printLog(pocl_text, "\n\nresult: failed\n");

        printLog(pocl_text, "\ndestroy opencl resources... ");
        destroyPoCL();
    }
}