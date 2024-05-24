package org.portablecl.poclaisademo;

public class DevelopmentVariables {


    /**
     * set how verbose the program should be.
     * 0 also disables pocl messages
     */
    public final static int VERBOSITY = 0;

    /**
     * if true, each function will print when they are being called
     */
    public static final boolean DEBUGEXECUTION = false;


    /**
     * allow the pocl image processor to fallback to local only on
     * a device lost
     */
    public static final boolean ENABLEFALLBACK = true;
}
