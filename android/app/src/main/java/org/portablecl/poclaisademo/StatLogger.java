package org.portablecl.poclaisademo;

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

    private long timeEnergy, timeBandw;
    private int amp, volt;
    private TrafficMonitor.DataPoint dataPoint;

    public StatLogger(FileOutputStream stream, TrafficMonitor trafficMonitor,
                      EnergyMonitor energyMonitor) {
        this.trafficMonitor = trafficMonitor;
        this.energyMonitor = energyMonitor;

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

            builder.append("time_bandw_ns, bandw_down_B, bandw_up_B, time_eng_ns, current_mA, " +
                    "voltage_mV\n");
            try {
                stream.write(builder.toString().getBytes());
            } catch (IOException e) {
                Log.println(Log.WARN, "Mainactivity.java:statlogger", "could not write csv " +
                        "header");
            }
        }
    }


    @Override
    public void run() {

        builder.setLength(0);

        dataPoint = trafficMonitor.pollTrafficStats();
        timeBandw = System.nanoTime();
        builder.append(timeBandw).append(",").append(dataPoint.rx_bytes_confirmed).append(",")
                .append(dataPoint.tx_bytes_confirmed).append(",");

        volt = energyMonitor.pollVoltage();
        amp = energyMonitor.pollCurrent();
        timeEnergy = System.nanoTime();
        builder.append(timeEnergy).append(",").append(amp).append(",").append(volt)
                .append("\n");

        if (null != stream) {
            try {
                stream.write(builder.toString().getBytes());
            } catch (IOException e) {
                Log.println(Log.WARN, "Mainactivity.java:statlogger", "could not write monitor " +
                        "data");
            }
        }

    }
}
