package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.DevelopmentVariables.VERBOSITY;
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

    /**
     * how many seconds between getting a response
     */
    private final float interval;

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
        this(ip, 0.2f);
    }

    /**
     * create a ping monitor that pings an ip
     * using the ping program on a subprocess.
     *
     * @param ip       address to ping
     * @param interval the polling interval of the ping in seconds. minimum is 0.2 seconds.
     */
    public PingMonitor(String ip, float interval) {

        if (interval < 0.2f) {
            throw new IllegalArgumentException("interval needs to be bigger than 200 ms");
        }
        totalPingTime = 0;
        pingCount = 0;
        ping = 0;
        IPAddress = ip;
        pingProcess = null;
        pingReader = null;
        this.interval = interval;

        pattern = Pattern.compile(regexPattern);

    }

    /**
     * start the ping subprocess and buffered reader.
     * needs to be called before getting ping stats.
     */
    public void start() {

        if(null != pingProcess || null != pingReader){
            if(VERBOSITY >= 1) {
                Log.println(Log.WARN, "pingmonitor", "ping monitor was already running when calling start");
            }
            return;
        }

        Runtime runtime = Runtime.getRuntime();
        try {
            pingProcess = runtime.exec("ping -i " + this.interval + " " + IPAddress);
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
     * blocks until the ping gives an update
     *
     * @return updated ping
     */
    public float blockingTick(int timeout) {
        Boolean gotPing = false;
        long endTime = System.currentTimeMillis() + timeout;
        try {
            String pingLine;
            while (!gotPing && System.currentTimeMillis() < endTime) {

                if (!pingReader.ready()) {
                    continue;
                }

                pingLine = pingReader.readLine();
                if (null == pingLine) {
                    break;
                }
                patternMatcher = pattern.matcher(pingLine);
                if (patternMatcher.find()) {
                    long timestamp = System.nanoTime();
                    ping = Float.parseFloat(Objects.requireNonNull(patternMatcher.group(2)));
                    totalPingTime += ping;
                    pingCount++;
                    gotPing = true;
                }
            }
        } catch (IOException e) {
            Log.println(Log.INFO, "pingmonitor", "encountered io exception reading");
            e.printStackTrace();
        } catch (Exception e) {
            Log.println(Log.INFO, "pingmonitor", "encountered exception reading ping");
            e.printStackTrace();
        }
        return ping;
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

    /**
     * creates a pingmonitor and destroys it after getting the result
     *
     * @param ipAddress to ping
     * @return the ping time
     */
    public static float singlePing(String ipAddress, int timeout) {
        PingMonitor monitor = new PingMonitor(ipAddress, 0.2f);
        monitor.start();
        float ping = monitor.blockingTick(timeout);
        monitor.stop();
        return ping;
    }

}
