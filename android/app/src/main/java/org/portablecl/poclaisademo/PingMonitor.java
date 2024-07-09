package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.JNIPoclImageProcessor.pushExternalPing;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PingMonitor {

    /**
     * regex to get the number of milliseconds the ping took
     */
    private final static String regexPattern = "(^.*=)([\\d\\.]+)(\\ .*$)";

    /**
     * sum of all pings so far
     */
    private float totalPingTime;

    /**
     * number of pings that have occurred so far
     */
    private int pingCount;

    /**
     * current ping time
     */
    private float ping;

    /**
     * address to ping
     */
    private final String IPAddress;

    /**
     * subprocess running the ping command
     */
    private Process pingProcess;

    /**
     * reader that gets the outputs from the ping command
     */
    private BufferedReader pingReader;

    /**
     * pattern matcher that has the matched line
     */
    private Matcher patternMatcher;

    /**
     * object to regex the read line
     */
    private final Pattern pattern;

    static {
        System.loadLibrary("poclnative");
    }

    /**
     * create a ping monitor that pings an ip
     * using the ping program on a subprocess.
     *
     * @param ip address to ping
     */
    public PingMonitor(String ip) {

        totalPingTime = 0;
        pingCount = 0;
        ping = 0;
        IPAddress = ip;

        pattern = Pattern.compile(regexPattern);

    }

    /**
     * start the ping subprocess and buffered reader.
     * needs to be called before getting ping stats.
     */
    public void start() {
        Runtime runtime = Runtime.getRuntime();
        try {
            pingProcess = runtime.exec("ping " + IPAddress);
            pingReader = new BufferedReader(new InputStreamReader(pingProcess.getInputStream()));

            // the first line is not relevant
            if (pingReader.ready()) {
                pingReader.readLine();
            }
        } catch (IOException e) {
            Log.println(Log.WARN, "pingmonitor", "error while trying to start pingMonitor");
            e.printStackTrace();
        }

    }

    /**
     * stop the ping subprocess and close the bufferred reader
     */
    public void stop() {
        try {
            if (null != pingReader) {
                pingReader.close();
                pingReader = null;
            }
            if (null != pingProcess) {
                pingProcess.destroy();
                pingProcess = null;
            }

        } catch (IOException e) {
            Log.println(Log.WARN, "pingmonitor", "error while trying to stop pingMonitor");
            e.printStackTrace();
        }
    }


    /**
     * reset the ping counts.
     */
    public void reset() {
        totalPingTime = 0;
        pingCount = 0;
        ping = 0;
    }

    public boolean isReaderNull() {
        return this.pingReader == null;
    }

    /**
     * function to flush the ping output and update the metrics.
     * gets called with getping and getaverageping
     */
    public void tick() {
        try {
            long timestamp = System.nanoTime();
            String pingLine;
            while (pingReader != null && pingReader.ready()) {

                pingLine = pingReader.readLine();
                if (null == pingLine) {
                    break;
                }
                patternMatcher = pattern.matcher(pingLine);
                if (patternMatcher.find()) {
                    ping = Float.parseFloat(Objects.requireNonNull(patternMatcher.group(2)));
                    totalPingTime += ping;
                    pingCount++;
                    pushExternalPing(timestamp, ping);
                }
            }
        } catch (IOException e) {
            Log.println(Log.INFO, "pingmonitor", "encountered io exception reading");
            e.printStackTrace();
        } catch (Exception e) {
            Log.println(Log.INFO, "pingmonitor", "encountered exception reading ping");
            e.printStackTrace();
        }
    }

    /**
     * returns the current ping value
     *
     * @return the current ping
     */
    public float getPing() {

        return ping;
    }

    /**
     * returns the average ping so far
     *
     * @return the average ping
     */
    public float getAveragePing() {
        if (0 == pingCount) {
            return 0;
        }
        return totalPingTime / pingCount;
    }

}
