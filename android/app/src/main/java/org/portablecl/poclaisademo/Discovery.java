package org.portablecl.poclaisademo;

import android.app.Activity;
import android.content.Context;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;
import android.net.wifi.WifiManager;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import android.widget.AdapterView;

public class Discovery {
    public Discovery(Activity activity){

        this.activity = activity;
        sNsdManager = (NsdManager) activity.getApplicationContext().getSystemService(Context.NSD_SERVICE);
    }
    private final Activity activity;
    private static NsdManager sNsdManager;
    private NsdManager.DiscoveryListener sDiscoveryListener;
    public static HashMap<String, serviceInfo> serviceMap;
    private static ArrayAdapter<String> spinnerAdapter;
    public static ArrayList<String> spinnerList;
    public static final String SERVICE_TYPE = "_pocl._tcp";
    public static final String TAG = "DISC";
    public static final String DEFAULT_SPINNER_VAL = "Select a server";
    public static native void addDevice(String key, int mode);
    private static WifiManager wifiManager;
    private static WifiManager.MulticastLock multicastLock;


    public static class serviceInfo {
        String name;
        int deviceCount;
        boolean reconnect;
    }

    private void initServiceMap(){
        destroyServiceMap();
        serviceMap = new HashMap<>();
    }
    private void destroyServiceMap(){
        if(serviceMap != null){
            serviceMap.clear();
            serviceMap = null;
        }
    }

    public void initDiscovery(Spinner discoverySpinner, AdapterView.OnItemSelectedListener listener) {

        spinnerList = new ArrayList<>();
        spinnerList.add(DEFAULT_SPINNER_VAL);
        spinnerAdapter = new ArrayAdapter<String>(activity, android.R.layout.simple_spinner_item, spinnerList);
        discoverySpinner.setAdapter(spinnerAdapter);
        discoverySpinner.setOnItemSelectedListener(listener);

        wifiManager = (WifiManager) activity.getSystemService(Context.WIFI_SERVICE);
        multicastLock = wifiManager.createMulticastLock(TAG);
        multicastLock.setReferenceCounted(true);
        multicastLock.acquire();

        stopDiscovery();
        initServiceMap();
        initializeDiscoveryListener();
        sNsdManager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, sDiscoveryListener);


    }
    public void stopDiscovery() {
        if (sDiscoveryListener != null) {
            try {
                sNsdManager.stopServiceDiscovery(sDiscoveryListener);
                sDiscoveryListener = null;
                multicastLock.release();
            }catch(Exception e){e.printStackTrace();}
        }
        destroyServiceMap();
    }

    private void initializeDiscoveryListener(){
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
                removeSpinnerEntry(nsdServiceInfo.getServiceName());
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
                Log.d(TAG, "Service resolved: " + nsdServiceInfo.getHost().toString() + ":" + nsdServiceInfo.getPort());
                insertService(nsdServiceInfo);
            }
        };
    }

    private void insertService(NsdServiceInfo nsdServiceInfo){
        String key = nsdServiceInfo.getHost().toString().substring(1) + ":" + nsdServiceInfo.getPort();
        String sName = nsdServiceInfo.getServiceName();
        String txt = nsdServiceInfo.getAttributes().toString();
        String substring = txt.substring(1, txt.length() - 6);

        if(serviceMap.containsKey(key)){
            if(serviceMap.get(key).name.equals(sName)){
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known with same session.");
//                serviceMap.get(key).reconnect = true;
//                addDevice(key, 1);
                addSpinnerEntry(key);
            }else {
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known but old session has expired.");
                serviceMap.get(key).name = sName;
                serviceMap.get(key).deviceCount = Integer.parseInt(substring);
                serviceMap.get(key).reconnect = false;

//                for(int i = 0; i < serviceMap.get(key).deviceCount; i++){
////                    addDevice(key + "/" + i,0);
//                }
                addSpinnerEntry(key);
            }
        }
        else {
            String _key = null;
            for(Map.Entry<String,serviceInfo> entry : serviceMap.entrySet()){
                if(entry.getValue().name.equals(sName)){
                    _key = entry.getKey();
                }
            }
            if(_key != null){
                Log.d(TAG, "(RESOLVER) Service " + sName + " is registered with a different address.");
            }else {
                serviceInfo found = new serviceInfo();
                found.name = sName;
                found.deviceCount = Integer.parseInt(substring);
                found.reconnect = false;
                serviceMap.put(key, found);

//                for(int i = 0; i < found.deviceCount; i++){
////                    addDevice(key + "/" + i,0);
//                }
                Log.d(TAG, "(RESOLVER) Service " + sName + "is being added.\n");
                addSpinnerEntry(key);

            }
        }
    }
    private void addSpinnerEntry(String spinnerText){
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if(!spinnerList.contains(spinnerText)) {
                    spinnerAdapter.add(spinnerText);
                    spinnerAdapter.notifyDataSetChanged();
                }
            }
        });
    }
    private void removeSpinnerEntry(String serviceName){
        String key = null;
        for(Map.Entry<String, Discovery.serviceInfo> entry : serviceMap.entrySet()) {
            if (Objects.equals(serviceName, entry.getValue().name)){
                key = entry.getKey();
                break;
            }
        }
        assert key!=null;
        String finalKey = key;
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                spinnerAdapter.remove(finalKey);
                spinnerAdapter.notifyDataSetChanged();
            }
        });
    }
}

