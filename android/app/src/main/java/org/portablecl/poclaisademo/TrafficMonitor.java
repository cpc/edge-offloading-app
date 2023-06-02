package org.portablecl.poclaisademo;


import static org.portablecl.poclaisademo.JNIPoclImageProcessor.getRemoteTrafficStats;

public class TrafficMonitor {
    static class DataPoint {
        public long tv_sec;
        public long tv_nsec;
        public long rx_bytes_requested;
        public long rx_bytes_confirmed;
        public long tx_bytes_submitted;
        public long tx_bytes_confirmed;

        public DataPoint(long tv_sec, long tv_nsec, long rx_bytes_requested, long rx_bytes_confirmed, long tx_bytes_submitted, long tx_bytes_confirmed) {
            this.tv_sec = tv_sec;
            this.tv_nsec = tv_nsec;
            this.rx_bytes_requested = rx_bytes_requested;
            this.rx_bytes_confirmed = rx_bytes_confirmed;
            this.tx_bytes_submitted = tx_bytes_submitted;
            this.tx_bytes_confirmed = tx_bytes_confirmed;
        }
    }

    private DataPoint prev;
    private DataPoint current;

    public TrafficMonitor() {
        current = new DataPoint(0, 0, 0, 0, 0, 0);
        prev = current;
    }

    public void reset() {
        current = new DataPoint(0, 0, 0, 0, 0, 0);
        prev = current;
    }

    public void tick() {
        prev = current;
        current = getRemoteTrafficStats();
    }

    private String magnitudeToPrefix(long magnitude) {
        if (magnitude == 1_000_000_000_000L)
            return "T";
        if (magnitude == 1_000_000_000L)
            return "G";
        if (magnitude == 1_000_000L)
            return "M";
        if (magnitude == 1_000L)
            return "k";
        return "";
    }

    private long getMagnitude(double val) {
        if (val > 1e12)
            return 1_000_000_000_000L;
        if (val > 1e9)
            return 1_000_000_000L;
        if (val > 1e6)
            return 1_000_000L;
        if (val > 1e3)
            return 1_000L;
        return 1;
    }

    final long NS_PER_S = 1_000_000_000L;

    public String getRXBandwidthString() {
        long delta_nsec = (current.tv_sec - prev.tv_sec) * NS_PER_S + (current.tv_nsec - prev.tv_nsec);
        double delta_s = (double) delta_nsec / (double) NS_PER_S;

        long delta_rx = current.rx_bytes_confirmed - prev.rx_bytes_confirmed;

        double bandwidth_rx = 8.0 * ((double) (delta_rx) / delta_s);

        long mag = getMagnitude(bandwidth_rx);
        bandwidth_rx /= (double) mag;

        return String.format("%6.1f %-4s", bandwidth_rx, magnitudeToPrefix(mag) + "bps");
    }

    public String getTXBandwidthString() {
        long delta_nsec = (current.tv_sec - prev.tv_sec) * NS_PER_S + (current.tv_nsec - prev.tv_nsec);
        double delta_s = (double) delta_nsec / (double) NS_PER_S;

        long delta_tx = current.tx_bytes_confirmed - prev.tx_bytes_confirmed;

        double bandwidth_tx = 8.0 * ((double) (delta_tx) / delta_s);

        long mag = getMagnitude(bandwidth_tx);
        bandwidth_tx /= (double) mag;

        return String.format("%6.1f %-4s", bandwidth_tx, magnitudeToPrefix(mag) + "bps");
    }
}
