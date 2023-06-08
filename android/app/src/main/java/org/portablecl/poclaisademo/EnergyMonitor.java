package org.portablecl.poclaisademo;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;


public class EnergyMonitor {

    /**
     * the android documentation says that BATTERY_PROPERTY_CURRENT_NOW
     * is in micro amps, however on samsung devices, this seems to be in milli amps.
     * doing some stopwatch testing confirms this.
     */
    private final float currentScale;
    /**
     * android documentation says that EXTRA_VOLTAGE is in
     * milli volts.
     */
    private static final float voltageScale = 0.001f;
    /**
     * the time is measured in nanoseconds so 1/10**9
     */
    private static final float timescale = 0.000000001f;
    /**
     * the battery manager used to query for values.
     */
    private final BatteryManager manager;
    /**
     * an intent used to listen to battery changes
     * It is used to query for voltage.
     */
    private final Intent receiver;
    /**
     * the total time since a reset or first tick.
     */
    private long totalTime;
    /**
     * the time when a previous tick happened.
     */
    private long previousTime;
    /**
     * the total net energy since a reset or first tick.
     */
    private float totalEnergyDelta;
    /**
     * the time since a tick happened.
     */
    private long timeFrame;
    /**
     * the net amount of energy since a tick.
     */
    private float energyFrame;
    /**
     * the current gotten from BATTERY_PROPERTY_CURRENT_NOW in milli amps
     */
    private int current;
    /**
     * the voltage gotten from EXTRA_VOLTAGE
     */
    private int voltage;

    /**
     * the smoothing factor determines how receptive the
     * ema_eps is to change. closer to 1 means very receptive.
     * finding this number is not a science, but this number
     * seems to work well for our use case
     */
    private final static float smoothing_factor = 0.3f;

    /**
     * an exponential moving average of the eps.
     * this eps does not jump as wildly.
     */
    private float ema_eps;

    public EnergyMonitor(Context context) {
        totalTime = 0;
        previousTime = 0;
        totalEnergyDelta = 0;
        timeFrame = 0;
        energyFrame = 0;
        ema_eps = 0;

        this.manager = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        // since battery changed is "sticky", we don't have to provide a receiver,
        // and this also means we don't have to unregister the receiver.
        this.receiver = context.registerReceiver(null,
                new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

        // a check to see if the property returns values in milli or micro amps
        if(manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) < 6000){
            currentScale = 0.001f;
        }else {
            currentScale = 0.000001f;
        }

    }

    /**
     * reset all values back to zero
     */
    public void reset() {
        totalTime = 0;
        previousTime = 0;
        totalEnergyDelta = 0;
        timeFrame = 0;
        energyFrame = 0;
        ema_eps = 0;
    }

    /**
     * Used to start a measurement. Can be called more often for finer measurements.
     */
    public void tick() {
        long currentTime = System.nanoTime();
        //https://developer.android.com/reference/android/os/BatteryManager
        current = manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
        // li-ion batteries are usually 3.7 volt, default to that if not available
        voltage = receiver.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 3700);

        if (previousTime == 0) {
            previousTime = currentTime;
            return;
        }

        timeFrame = currentTime - previousTime;
        previousTime = currentTime;
        totalTime += timeFrame;
        // joule = coulomb * volt
        // coulomb = 1 amp for 1 second
        // joule = current * time * volt
        energyFrame = (current * currentScale) * (voltage * voltageScale);
        totalEnergyDelta += energyFrame * (timeFrame * timescale);

        ema_eps = ema_eps + smoothing_factor*((energyFrame / (timeFrame * timescale)) - ema_eps);

    }

    /**
     * Return the current charge left in the battery.
     * while this is more accurate than the default battery gauge,
     * it is not accurate enough for us.
     * This function is more for debugging.
     *
     * @return the charge left in micro Amperehours (5000,000 is a full charge)
     */
    public int getCharge() {
        return manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
    }

    /**
     * get the voltage that android reads
     *
     * @return the voltage in milli volts
     */
    public int getVoltage() {
        return voltage;
    }

    /**
     * get the net current that is flowing through the battery
     *
     * @return the current in milli amps, negative numbers indicate a drainage.
     */
    public int getcurrent() {
        return current;
    }

    /**
     * get the energy per second drained from the battery in joules per second
     *
     * @return the net joules per second leaving the battery
     */
    public float getEPS() {
        if (timeFrame == 0) {
            return 0;
        }

        return energyFrame / (timeFrame * timescale);
    }

    /**
     * reutrn the EMA FPS
     * @return exponential moving average energy per second
     */
    public float getEMAEPS(){
        return ema_eps;
    }

    /**
     * get the total average energy per second drained since starting to measure.
     *
     * @return the average
     */
    public float getAverageEPS() {
        if (totalTime == 0) {
            return 0;
        }

        return (totalEnergyDelta / (totalTime * timescale));
    }
}
