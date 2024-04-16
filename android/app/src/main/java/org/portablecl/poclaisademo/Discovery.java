package org.portablecl.poclaisademo;

import android.content.Context;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;

//import java.security.SecureRandom;
//import java.util.Arrays;

public class Discovery {

    public static native void addDevice(String key, int mode);

    NsdManager sNsdManager;
    NsdManager.ResolveListener sResolverListener;
    NsdManager.DiscoveryListener sDiscoveryListener;

    public static final String SERVICE_TYPE = "_pocl._tcp";
    public static final String TAG = "DISC";

    public HashMap<String, serviceInfo> serviceMap;
    String serviceName;
    public static class serviceInfo {
        String name;
        int deviceCount;
        boolean newService;
        boolean reconnect;
    }

    private void initServiceMap(){
        destroyServiceMap();
        serviceMap = new HashMap<>();
    }

    private void insertService(NsdServiceInfo nsdServiceInfo){
        String key = nsdServiceInfo.getHost().toString() + ":" + nsdServiceInfo.getPort();
        key = key.substring(1);
        String sName = nsdServiceInfo.getServiceName();
        String txt = nsdServiceInfo.getAttributes().toString();
        String substring = txt.substring(1, txt.length() - 6);

        if(serviceMap.containsKey(key)){
            if(serviceMap.get(key).name.equals(sName)){
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known with same session.");
                serviceMap.get(key).newService = false;
                serviceMap.get(key).reconnect = true;
                addDevice(key, 1);
            }else {
                Log.d(TAG, "(RESOLVER) Service " + sName + " is known but old session has expired.");
                serviceMap.get(key).name = sName;
                serviceMap.get(key).deviceCount = Integer.parseInt(substring);
                serviceMap.get(key).newService = true;
                serviceMap.get(key).reconnect = false;

                for(int i = 0; i < serviceMap.get(key).deviceCount; i++){
                    addDevice(key + "/" + i,0);
                }
                //Log.d(TAG, serviceMap.keySet().toString());
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
                found.newService = true;
                found.reconnect = false;
                serviceMap.put(key, found);

                for(int i = 0; i < found.deviceCount; i++){
                    addDevice(key + "/" + i,0);
                }
                Log.d(TAG, "(RESOLVER) Service " + sName + "is being added.\n");
            }
        }
    }
    private void destroyServiceMap(){
        if(serviceMap != null){
            serviceMap.clear();
            serviceMap = null;
        }
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
                sNsdManager.resolveService(nsdServiceInfo, sResolverListener);
            }

            @Override
            public void onServiceLost(NsdServiceInfo nsdServiceInfo) {
                Log.d(TAG, "Service lost: " + nsdServiceInfo);
            }
        };
    }

    private void initializeResolveListener() {
        sResolverListener = new NsdManager.ResolveListener() {
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
    public void initDiscovery(Context context) {
        stopDiscovery();
        initServiceMap();
        initializeDiscoveryListener();
        initializeResolveListener();
        sNsdManager = (NsdManager) context.getSystemService(Context.NSD_SERVICE);
        sNsdManager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, sDiscoveryListener);
    }

    public void stopDiscovery() {
        if (sDiscoveryListener != null) {
            try {
                sNsdManager.stopServiceDiscovery(sDiscoveryListener);
            }finally {
            }
            sDiscoveryListener = null;
        }
        destroyServiceMap();
    }

}

