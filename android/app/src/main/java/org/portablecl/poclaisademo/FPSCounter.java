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
     * the time is measured in nanoseconds so 10**9
     */
    private final float timePrecision = 1000000000f;

    public FPSCounter() {
        totalTime = 0;
        frameCount = 0;
        frameTime = 0;
        previousFrameTime = 0;
    }

    /**
     * set all metric values to zero
     */
    public void reset() {
        totalTime = 0;
        frameCount = 0;
        frameTime = 0;
        previousFrameTime = 0;
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

        totalTime += frameTime;
        frameCount += 1;

        previousFrameTime = currentTime;

        return;
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
