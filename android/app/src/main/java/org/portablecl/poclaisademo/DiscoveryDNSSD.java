package org.portablecl.poclaisademo;

import android.util.Log;

import org.minidns.dnsname.DnsName;
import org.minidns.hla.ResolverApi;
import org.minidns.hla.ResolverResult;
import org.minidns.hla.SrvResolverResult;
import org.minidns.record.A;
import org.minidns.record.PTR;
import org.minidns.record.SRV;
import org.minidns.record.TXT;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Executor;

/**
 * This class implements DNS-SD service discovery and adds found services through DiscoverySelect
 * reference.
 */
public class DiscoveryDNSSD {

    private Executor executor;
    private Set<PTR> foundServices = new HashSet<PTR>();
    public String domain;
    public String type;

    private final DiscoverySelect DS;

    /**
     * @param domain domain corresponds the domain in which DNS-SD services are needed to be
     *               serached
     * @param type   type corresponds to the service type. eg: _pocl._tcp, _ipp._tcp
     * @param ds     reference to DiscoverySelect class object
     */
    DiscoveryDNSSD(String domain, String type, DiscoverySelect ds) {
        this.type = type;
        this.domain = domain;
        this.DS = ds;
    }

    /**
     * Search for services. If found, they'll be added to the spinner through DiscoverySelect object
     */
    public void getDNSSDService() {
        serviceBrowser();
    }

    /**
     * Remove all services found through DNS-SD from the spinner
     */
    public void removeServices() {
        if (!foundServices.isEmpty()) {
            for (PTR srv : foundServices) {
                DS.removeSpinnerEntry(srv.toString().substring(0, 32));
            }
        }
    }

    /**
     * This function is responsible for conducting the search, it finds all available services
     * registered under the provided service type. The services are then resolved to get IPs,
     * ports, and txt records. These are then used to insert the services to the spinner and
     * serviceMap.
     */
    private void serviceBrowser() {
        Set<PTR> services = getTypeServices();
        if (null != services) {
            foundServices.addAll(services);

            for (PTR srv : services) {
                String sName = null;
                String key = null;
                String txt = null;

                String service = srv.toString();
                sName = service.substring(0, 32);
                Set<SRV> result = resolveSRV(service);
                txt = resolveTXT(service);

                if (null != result && null != txt) {
                    for (SRV items : result) {
                        String add = resolveA(items.target);
                        if (null == add) break;
                        key = add.substring(1) + ":" + items.port;

                        Log.d("DISC",
                                "DNS-SD found service: " + sName + ", IPv4: " + key + ", " + "TXT" +
                                        ":" + txt);
                        DS.insertService(sName, key, txt);
                        break;
                    }
                }

            }
        }
    }

    private Set<PTR> getTypeServices() {

        ResolverResult<PTR> ptrBrowseDomain = resolvePTR("b._dns-sd._udp." + domain, "No " +
                "browse domain found in the domain: " + domain);
        if (null == ptrBrowseDomain) return null;
        String browseDomain = ptrBrowseDomain.getAnswers().toArray()[0].toString();

        ResolverResult<PTR> ptrServiceTypes = resolvePTR("_services._dns-sd._udp." + browseDomain
                , "DNS-SD services " + "not available in the domain:  " + domain);
        if (null == ptrServiceTypes) return null;
        String serviceType = null;
        for (PTR types : ptrServiceTypes.getAnswers()) {
            if (types.toString().contains(type)) {
                serviceType = types.toString();
                break;
            }
        }
        if (null == serviceType) {
            Log.w("DISC", type + " type services not found in the domain:  " + domain);
            return null;
        }

        ResolverResult<PTR> ptrServices = resolvePTR(serviceType, "Instances of " + type + " " +
                "not found in the domain:  " + domain);
        if (null == ptrServices) return null;
        return ptrServices.getAnswers();
    }

    private ResolverResult<PTR> resolvePTR(String name, String logMessage) {
        try {
            ResolverResult<PTR> result = ResolverApi.INSTANCE.resolve(name, PTR.class);
            if (null == result || !result.wasSuccessful()) {
                Log.e("DISC", logMessage);
                return null;
            }
            return result;
        } catch (IOException e) {
            Log.e("DISC", "failed IO", e);
            return null;
        } catch (Throwable e) {
            Log.e("DISC", "failed", e);
            return null;
        }
    }

    private Set<SRV> resolveSRV(String name) {
        try {
            SrvResolverResult result = ResolverApi.INSTANCE.resolveSrv(name);
            if (null == result || !result.wasSuccessful()) {
                return null;
            }
            return result.getAnswers();
        } catch (IOException e) {
            Log.e("DISC", "failed IO", e);
            return null;
        } catch (Throwable e) {
            Log.e("DISC", "failed", e);
            return null;
        }
    }

    private String resolveA(DnsName name) {
        try {
            ResolverResult<A> result = ResolverApi.INSTANCE.resolve(name, A.class);
            if (null == result || !result.wasSuccessful()) {
                Log.e("DISC", "Could not resolve " + name + " for a IPv4 address.");
                return null;
            }
            String add = null;
            for (A items : result.getAnswers()) {
                add = items.getInetAddress().toString();
                break;
            }
            return add;
        } catch (IOException e) {
            Log.e("DISC", "failed IO", e);
            return null;
        } catch (Throwable e) {
            Log.e("DISC", "failed", e);
            return null;
        }
    }

    private String resolveTXT(String name) {
        try {
            ResolverResult<TXT> result = ResolverApi.INSTANCE.resolve(name, TXT.class);
            if (null == result || !result.wasSuccessful()) {
                Log.e("DISC", "Could not get TXT record for ");
                return null;
            }
            String txt = null;
            for (TXT items : result.getAnswers()) {
                txt = items.getText();
                break;
            }
            return txt;
        } catch (IOException e) {
            Log.e("DISC", "failed IO", e);
            return null;
        } catch (Throwable e) {
            Log.e("DISC", "failed", e);
            return null;
        }
    }
}
