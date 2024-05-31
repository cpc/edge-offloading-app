package org.portablecl.poclaisademo;

import android.app.Activity;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.nsd.NsdServiceInfo;
import android.util.Log;
import android.widget.AdapterView;
import android.widget.Spinner;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/*
 * This class implements the spinner, initializes network discovery, implements required
 * sub-classes serviceInfo and spinnerObject, implements methods to add and remove discovered
 * devices.
 */
public class DiscoverySelect {

    public class serviceInfo {
        String name;
        int deviceCount;
        boolean reconnect;
    }

    public class spinnerObject {
        public PingMonitor pingMonitor;
        public float ping;
        public String address;
        public String device_type;

        public spinnerObject() {
            this.address = DEFAULT_SPINNER_VAL;
        }

        public spinnerObject(String address, String device_type) {
            this.address = address;
            this.device_type = device_type;

            pingMonitor = new PingMonitor(address.split(":")[0]);
            pingMonitor.start();
            pingMonitor.reset();
            this.ping = pingMonitor.getPing();
        }

        public void updatePing() {
            this.ping = pingMonitor.getPing();
        }

        public String getDescription() {
            return ("(" + ping + "ms) " + " | " + device_type + " | " + address);
        }

        public String getAddress() {
            return (address);
        }

        public void Destroy() {
            pingMonitor.reset();
            pingMonitor.stop();
        }
    }

    private final Activity activity;
    public HashMap<String, serviceInfo> serviceMap;
    private final DiscoverySpinnerAdapter spinnerAdapter;
    private static Discovery NSDiscovery;
    public ArrayList<spinnerObject> spinnerList;
    public static final String TAG = "DISC";
    public static final String DEFAULT_SPINNER_VAL = "Select a server";
    public final spinnerObject verneServer = new spinnerObject("10.1.200.5:10998", "GPU");

    public DiscoverySelect(Activity activity, Spinner discoverySpinner,
                           AdapterView.OnItemSelectedListener listener) {
        this.activity = activity;
        destroyServiceMap();
        NSDiscovery = new Discovery();
        spinnerList = new ArrayList<>();
        spinnerList.add(new spinnerObject());
        spinnerAdapter = new DiscoverySpinnerAdapter(activity,
                android.R.layout.simple_spinner_item, spinnerList);
        discoverySpinner.setAdapter(spinnerAdapter);
        discoverySpinner.setOnItemSelectedListener(listener);
        initServiceMap();
        initVerneServer();
        NSDiscovery.initDiscovery(this, activity);
    }

    public void stopDiscovery() {
        NSDiscovery.stopDiscovery();
        destroyServiceMap();
    }

    private void initServiceMap() {
        destroyServiceMap();
        serviceMap = new HashMap<>();
    }

    private void destroyServiceMap() {
        if (serviceMap != null) {
            serviceMap.clear();
            serviceMap = null;
        }
    }

    // The verne server details are dynamically added to the spinner as and when the network
    // status is changed from wifi to cellular
    // TODO: Hardcoded parts to be modified when WAN discovery is implemented
    private void initVerneServer() {
        serviceInfo verne = new serviceInfo();
        verne.name = "cpu-icelake-server-Intel(R) Xeon";
        verne.deviceCount = 2;
        verne.reconnect = false;
        serviceMap.put("10.1.200.5:10998", verne);
        ConnectivityManager connectivityManager =
                activity.getSystemService(ConnectivityManager.class);
        connectivityManager.registerDefaultNetworkCallback(new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(@NonNull Network network) {
            }

            @Override
            public void onCapabilitiesChanged(@NonNull Network network,
                                              @NonNull NetworkCapabilities networkCapabilities) {
                if (networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)) {
                    if (!spinnerList.contains(verneServer)) {
                        spinnerList.add(verneServer);
                    }
                } else {
                    spinnerList.remove(verneServer);
                }
            }
        });
    }

    public void insertService(NsdServiceInfo nsdServiceInfo) {
        String key =
                nsdServiceInfo.getHost().toString().substring(1) + ":" + nsdServiceInfo.getPort();
        String sName = nsdServiceInfo.getServiceName();
        // NSD discovery adds '{' before the txt sent from the server and adds '=null}' after the
        // txt. These have to be accounted for when using the txt field.
        String txt = nsdServiceInfo.getAttributes().toString();
        int devices = txt.length() - 7;
        String deviceType;
        // 0:CL_DEVICE_TYPE_CPU , 1:CL_DEVICE_TYPE_GPU , 2:CL_DEVICE_TYPE_ACCELERATOR ,
        // 4:CL_DEVICE_TYPE_CUSTOM
        switch (txt.substring(1, 2)) {
            case "0":
                deviceType = "CPU";
                break;
            case "1":
                deviceType = "GPU";
                break;
            case "2":
                deviceType = "Accelerator";
                break;
            case "4":
                deviceType = "Custom";
                break;
            default:
                deviceType = "NA";
        }

        // logic to decide if a discovered server is new or old
        if (serviceMap.containsKey(key)) {
            if (serviceMap.get(key).name.equals(sName)) {
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known with same session.");
                addSpinnerEntry(new spinnerObject(key, deviceType));
            } else {
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known but old session has expired" +
                        ".");
                serviceMap.get(key).name = sName;
                serviceMap.get(key).deviceCount = devices;
                serviceMap.get(key).reconnect = false;
                addSpinnerEntry(new spinnerObject(key, deviceType));
            }
        } else {
            String _key = null;
            for (Map.Entry<String, serviceInfo> entry : serviceMap.entrySet()) {
                if (entry.getValue().name.equals(sName)) {
                    _key = entry.getKey();
                }
            }
            if (_key != null) {
                Log.d(TAG, "(RESOLVER) Service " + sName + " is registered with a different " +
                        "address.");
            } else {
                serviceInfo found = new serviceInfo();
                found.name = sName;
                found.deviceCount = devices;
                found.reconnect = false;
                serviceMap.put(key, found);
                Log.d(TAG, "(RESOLVER) Service " + sName + "is being added.\n");
                addSpinnerEntry(new spinnerObject(key, deviceType));
            }
        }
    }

    public void addSpinnerEntry(spinnerObject so) {

        boolean contains = false;
        for (spinnerObject value : spinnerList) {
            if (value.address.contains(so.address)) {
                contains = true;
                break;
            }
        }
        boolean finalContains = contains;
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {

                if (!finalContains) {
                    spinnerList.add(so);
                    spinnerAdapter.notifyDataSetChanged();
                }
            }
        });

    }

    public void removeSpinnerEntry(String serviceName) {
        String key = null;
        for (Map.Entry<String, serviceInfo> entry : serviceMap.entrySet()) {
            if (Objects.equals(serviceName, entry.getValue().name)) {
                try {
                    key = entry.getKey();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            }
        }
        if (key == null) {
            Log.w(TAG, "Key is null. Spinner entry " + serviceName + " previously removed.");
            return;
        }
        String finalKey = key;
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {

                for (spinnerObject value : spinnerList) {
                    if (value.address.contains(finalKey)) {
                        value.Destroy();
                        spinnerList.remove(value);
                        spinnerAdapter.notifyDataSetChanged();
                        break;
                    }
                }
            }
        });
    }
}
