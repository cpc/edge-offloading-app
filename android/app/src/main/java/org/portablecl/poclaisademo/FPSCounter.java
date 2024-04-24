package org.portablecl.poclaisademo;

/**
 * A class that keeps and calculates metrics related to Frames Per Second (FPS).
 */
public class FPSCounter {

    private long totalTime;

    private long frameCount;

    private long frameTime;

    private long previousFrameTime;

    /**
     * an exponential moving average of the fps.
     * this fps does not jump as wildly.
     */
    private float ema_fps;

    long timespanPrevFrameTime;
    long timespanFrameCount;

    float timespanEmaFps;

    /**
     * the smoothing factor determines how receptive the
     * ema_fps is to change. closer to 1 means very receptive.
     * finding this number is not a science, but this number
     * seems to work well for our use case
     */
    private final static float smoothingFactor = 0.6f;

    /**
     * the time is measured in nanoseconds so 10**9
     */
    private final float timePrecision = 1000000000f;

    public FPSCounter() {
        totalTime = 0;
        frameCount = 0;
        frameTime = 0;
        previousFrameTime = 0;
        ema_fps = 0;

        timespanPrevFrameTime = 0;
        timespanFrameCount = 0;
        timespanEmaFps = 0;
    }

    /**
     * set all metric values to zero
     */
    public void reset() {
        totalTime = 0;
        frameCount = 0;
        frameTime = 0;
        previousFrameTime = 0;
        ema_fps = 0;

        timespanPrevFrameTime = 0;
        timespanFrameCount = 0;
        timespanEmaFps = 0;
    }

    /**
     * measures times between being called to calculate FPS.
     * Should be called after every frame has been shown
     */
    public void tickFrame() {
        long currentTime = System.nanoTime();

        if (previousFrameTime == 0) {
            previousFrameTime = currentTime;

            return;
        }

        frameTime = currentTime - previousFrameTime;

        ema_fps = ema_fps + smoothingFactor * ((timePrecision / frameTime) - ema_fps);

        totalTime += frameTime;
        frameCount += 1;

        previousFrameTime = currentTime;

    }

    /**
     * Get the current FPS
     *
     * @return the current FPS as a float
     */
    public float getFPS() {
        // avoid those pesky division by zero errors
        if (frameTime == 0) {
            return 0;
        }
        return timePrecision / frameTime;
    }

    /**
     * return the EMA FPS
     *
     * @return the Exponential Moving Average Frames per Second
     */
    public float getEMAFPS() {
        return ema_fps;
    }

    /**
     * return the exponential moving average fps over the timespan
     * this function was last called
     *
     * @return
     */
    public float getEMAFPSTimespan() {

        long curTime = System.nanoTime();
        if (timespanPrevFrameTime == 0) {
            timespanPrevFrameTime = curTime;
            timespanFrameCount = frameCount;
            return 0;
        }

        long frameTime = curTime - timespanPrevFrameTime;
        timespanPrevFrameTime = curTime;

        long curFrameCount = frameCount;
        long framediff = curFrameCount - timespanFrameCount;
        timespanFrameCount = curFrameCount;

        float fps = (framediff / (frameTime / timePrecision));
        timespanEmaFps = timespanEmaFps + smoothingFactor * (fps - timespanEmaFps);

        return timespanEmaFps;
    }

    /**
     * Get the average FPS since TickFrame() started being called
     *
     * @return average FPS as a float
     */
    public float getAverageFPS() {
        if (totalTime == 0) {
            return 0;
        }
        return (frameCount * timePrecision) / totalTime;
    }


}
