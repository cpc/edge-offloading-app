package org.portablecl.poclaisademo;

import android.app.Activity;
import android.content.Context;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;
import android.net.wifi.WifiManager;
import android.util.Log;

/*
 * Class implements necessary methods to utilize the android NSD service for network discovery using
 * mDNS & DNS-SD.
 */
public class Discovery {
    private NsdManager sNsdManager;
    private NsdManager.DiscoveryListener sDiscoveryListener;
    public static final String SERVICE_TYPE = "_pocl._tcp";
    public static final String TAG = "DISC";
    /*
     * This instance of DiscoverySelect and the native function addDevice are the only
     * non-generic addition to this class. If this class is required to be used for other purpose
     *  than remote server discovery then only this part may be modified.
     */
    DiscoverySelect DS;

    public static native void addDevice(String key, int mode);


    public void initDiscovery(DiscoverySelect ds, Activity activity) {
        sNsdManager =
                (NsdManager) activity.getApplicationContext().getSystemService(Context.NSD_SERVICE);
        DS = ds;
        WifiManager wifiManager = (WifiManager) activity.getSystemService(Context.WIFI_SERVICE);

        stopDiscovery();
        initializeDiscoveryListener();
        sNsdManager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, sDiscoveryListener);
    }

    public void stopDiscovery() {
        if (sDiscoveryListener != null) {
            try {
                sNsdManager.stopServiceDiscovery(sDiscoveryListener);
                sDiscoveryListener = null;
                sNsdManager = null;
                DS = null;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void initializeDiscoveryListener() {
        sDiscoveryListener = new NsdManager.DiscoveryListener() {
            @Override
            public void onStartDiscoveryFailed(String s, int i) {
                Log.e(TAG, "Discovery failed: Error code: " + i);
            }

            @Override
            public void onStopDiscoveryFailed(String s, int i) {
                Log.e(TAG, "Discovery failed: Error code: " + i);
            }

            @Override
            public void onDiscoveryStarted(String s) {
                Log.d(TAG, "Discovery started:  " + s);
            }

            @Override
            public void onDiscoveryStopped(String s) {
                Log.d(TAG, "Discovery stopped:  " + s);
            }

            @Override
            public void onServiceFound(NsdServiceInfo nsdServiceInfo) {
                Log.d(TAG, "Service found: " + nsdServiceInfo);
                sNsdManager.resolveService(nsdServiceInfo, initializeResolveListener());
            }

            @Override
            public void onServiceLost(NsdServiceInfo nsdServiceInfo) {
                Log.d(TAG, "Service lost: " + nsdServiceInfo);
                // We use the NSD thread itself to perform appropriate method when a service is lost
                DS.removeSpinnerEntry(nsdServiceInfo.getServiceName());
            }
        };
    }

    private NsdManager.ResolveListener initializeResolveListener() {
        return new NsdManager.ResolveListener() {
            @Override
            public void onResolveFailed(NsdServiceInfo nsdServiceInfo, int i) {
                Log.e(TAG, "Resolve failed: Error code" + i);
            }

            @Override
            public void onServiceResolved(NsdServiceInfo nsdServiceInfo) {
                Log.d(TAG,
                        "Service resolved: " + nsdServiceInfo.getHost().toString() + ":" + nsdServiceInfo.getPort());
                // We use the NSD thread to perform appropriate method when a new service is found
                DS.insertService(nsdServiceInfo);
            }
        };
    }
}

