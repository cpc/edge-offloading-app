package org.portablecl.poclaisademo;

import static org.portablecl.poclaisademo.JNIPoclImageProcessor.pushExternalPing;
import static org.portablecl.poclaisademo.JNIPoclImageProcessor.pushExternalPow;

import android.util.Log;

import java.io.FileOutputStream;
import java.io.IOException;

/**
 * a custom implementation of a runnable that logs the values normally used to display
 * overlay results.
 */
public class StatLogger implements Runnable {

    private final StringBuilder builder;
    private FileOutputStream stream;

    private final TrafficMonitor trafficMonitor;

    private final EnergyMonitor energyMonitor;

    private PingMonitor pingMonitor;

    private int amp, volt;
    private int prev_amp, prev_volt;

    private float prev_ping_ms;

    public StatLogger(FileOutputStream stream, TrafficMonitor trafficMonitor,
                      EnergyMonitor energyMonitor, PingMonitor pingMonitor) {
        this.trafficMonitor = trafficMonitor;
        this.energyMonitor = energyMonitor;
        this.pingMonitor = pingMonitor;
        this.prev_amp = 0;
        this.prev_volt = 0;

        builder = new StringBuilder();

        setStream(stream);
    }

    /**
     * query the energymonitor to calculate current energy usage
     * @return
     */
    public float getCurrentEnergy() {
        return energyMonitor.calculateEnergy(volt, amp);
    }

    /**
     * set the file outputstream and write the csv header if the stream is not null
     *
     * @param stream
     */
    public void setStream(FileOutputStream stream) {
        this.stream = stream;

        if (null != this.stream) {

            builder.append("time_bandw_ns,bandw_down_B,bandw_up_B,time_eng_ns,current_mA," +
                    "voltage_mV,ping_ms\n");
            try {
                stream.write(builder.toString().getBytes());
            } catch (IOException e) {
                Log.println(Log.WARN, "Mainactivity.java:statlogger", "could not write csv " +
                        "header");
            }
        }
    }

    public void setPingMonitor(PingMonitor pingMonitor) {
        this.pingMonitor = pingMonitor;
    }


    @Override
    public void run() {
        TrafficMonitor.DataPoint dataPoint = trafficMonitor.pollTrafficStats();
        long timeBandw = System.nanoTime();

        volt = energyMonitor.pollVoltage();
        amp = energyMonitor.pollCurrent();
        long timeEnergy = System.nanoTime();

        // Do not push repeated samples which skew the statistics
        if (amp != prev_amp || volt != prev_volt) {
            pushExternalPow(timeEnergy, amp, volt);
            prev_amp = amp;
            prev_volt = volt;
        }

        float ping_ms = -1.0f;
//        if (null != pingMonitor) {
//            long timePing = System.nanoTime();
//            pingMonitor.tick();
//            ping_ms = pingMonitor.getPing();
//
//            // Do not push repeated samples which skew the statistics
//            if (ping_ms != prev_ping_ms) {
//                prev_ping_ms = ping_ms;
//                pushExternalPing(timePing, ping_ms);
//            }
//        }

        if (null != stream) {
            builder.setLength(0);
            builder.append(timeBandw).append(",").append(dataPoint.rx_bytes_confirmed).append(",")
                    .append(dataPoint.tx_bytes_confirmed);
            builder.append(",").append(timeEnergy).append(",").append(amp).append(",").append(volt);
            builder.append(",").append(ping_ms);
            builder.append("\n");

            try {
                stream.write(builder.toString().getBytes());
            } catch (IOException e) {
                Log.println(Log.WARN, "Mainactivity.java:statlogger", "could not write monitor " +
                        "data");
            }
        }

    }
}
