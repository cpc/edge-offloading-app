#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "discovery.h"
#include "pocl_debug.h"
#include "pocl_runtime_config.h"
#include "uthash.h"

#include <avahi-client/lookup.h>
#include <avahi-common/domain.h>
#include <avahi-common/error.h>
#include <avahi-common/malloc.h>
#include <avahi-common/thread-watch.h>

cl_int (*discovery_cb)(char *const, unsigned);
unsigned dev_typ;
cl_int (*pocl_remote_reconnect)(const char *);

// Struct to store service information
typedef struct ServiceInfo {
  char *sName; // Service name, has to be less than 63 characters - specified in
               // AVAHI doxygen
  char *sType; //_pocl._tcp.local
  char *sDomainName;    // Domain name 'local' for mDNS
  char *sAddr;          // IP address with port -> addr:port
  uint16_t DeviceCount; // Number of devices in the platform
  UT_hash_handle hh;    // Hashable structure
} ServiceInfo;

#define FREE_INFO(__OBJ__)                                                     \
  do {                                                                         \
    free(__OBJ__->sName);                                                      \
    free(__OBJ__->sType);                                                      \
    free(__OBJ__->sDomainName);                                                \
    free(__OBJ__->sAddr);                                                      \
  } while (0)

void DestroyServiceInfoTable(ServiceInfo *InfoPointer) {
  ServiceInfo *CurrInfo;
  ServiceInfo *Temp;

  HASH_ITER(hh, InfoPointer, CurrInfo, Temp) {
    FREE_INFO(CurrInfo);
    HASH_DEL(InfoPointer, CurrInfo);
    free(CurrInfo);
  }
}

ServiceInfo *ServiceInfoTable = NULL;

// Size decided based on this format:
// POCL_REMOTEX_PARAMETERS=hostname[:port]/INDEX
#define MAX_PARAM_LEN AVAHI_ADDRESS_STR_MAX + 1 + 5 + 1 + 5

AvahiThreadedPoll *thread_poll = NULL;
AvahiClient *client = NULL;
AvahiServiceBrowser **sb = NULL;
int domain_count;

int register_server(ServiceInfo *S, const char *name, const char *type,
                    const char *domain, AvahiStringList *txt, const char *key) {
  // TODO: Request for TXT record incase lost and not present here. Number of
  // devices and filteration in Remote driver depends on TXT records
  if (NULL != txt) {
    ServiceInfo *server = S;
    if (NULL == server) {
      server = (ServiceInfo *)calloc(1, sizeof(*server));
      server->sName = strndup(name, AVAHI_LABEL_MAX);
      server->sType = strndup(type, AVAHI_DOMAIN_NAME_MAX);
      server->sDomainName = strndup(domain, AVAHI_DOMAIN_NAME_MAX);
      server->sAddr = strndup(key, MAX_PARAM_LEN);
    }

    char *text = avahi_string_list_to_string(txt);
    // avahi_string_list_to_string return text field enclosed in ""
    server->DeviceCount = strlen(text) - 2;
    avahi_free(text);

    for (int i = 0; i < server->DeviceCount; i++) {
      char parameters[MAX_PARAM_LEN];
      snprintf(parameters, sizeof(parameters), "%s/%d", server->sAddr, i);
      int error = discovery_cb(parameters, dev_typ);
      if (0 != error) {
        if (NULL == S) {
          FREE_INFO(server);
          free(server);
        }
        POCL_MSG_ERR(
            "(RESOLVER) Device couldn't be added, skipping this server. \n");
        return error;
      }
    }

    if (NULL == S)
      HASH_ADD_KEYPTR(hh, ServiceInfoTable, server->sAddr,
                      strlen(server->sAddr), server);
  } else {
    POCL_MSG_ERR("TXT field not recieved through discovery, unknown number of "
                 "devices in remote server!");
    return -1;
  }
  return 0;
}

/* Called whenever a service has been resolved successfully or timed out */
static void resolve_callback(AvahiServiceResolver *r,
                             AVAHI_GCC_UNUSED AvahiIfIndex interface,
                             AVAHI_GCC_UNUSED AvahiProtocol protocol,
                             AvahiResolverEvent event, const char *name,
                             const char *type, const char *domain,
                             const char *host_name, const AvahiAddress *address,
                             uint16_t port, AvahiStringList *txt,
                             AvahiLookupResultFlags flags,
                             AVAHI_GCC_UNUSED void *userdata) {
  assert(r);

  ServiceInfo *head = ServiceInfoTable, *S = NULL;

  char a[AVAHI_ADDRESS_STR_MAX];
  char key[MAX_PARAM_LEN];

  switch (event) {
  case AVAHI_RESOLVER_FAILURE: {
    POCL_MSG_ERR("(Resolver) Failed to resolve service '%s' of type '%s' in "
                 "domain '%s': %s\n",
                 name, type, domain,
                 avahi_strerror(
                     avahi_client_errno(avahi_service_resolver_get_client(r))));
    break;
  }
  case AVAHI_RESOLVER_FOUND: {
    /*
        address && name -> reconnect normally
        !address && name -> ignore new address
        address && !name -> session changed, add as new service but need to
       handle repeatr address in find or create new server function !address &&
       !name -> add as a new service
    */

    avahi_address_snprint(a, sizeof(a), address);
    snprintf(key, sizeof(key), "%s:%d", a, port);
    HASH_FIND_STR(head, key, S);

    if (NULL != S) {

      if (!strcmp(S->sName, name)) {
        POCL_MSG_PRINT_REMOTE("(RESOLVER) Server '%s' of type '%s' in domain "
                              "'%s' is known with same session. \n",
                              name, type, domain);
        pocl_remote_reconnect(key);
      } else {
        POCL_MSG_PRINT_REMOTE("(RESOLVER) Server '%s' of type '%s' in domain "
                              "'%s' is known, but old session has expired. \n",
                              name, type, domain);
        register_server(S, name, type, domain, txt, key);
      }
    } else {
      for (S = ServiceInfoTable; S; S = S->hh.next) {
        if (!strcmp(S->sName, name))
          break;
      }

      if (NULL != S) {
        POCL_MSG_PRINT_REMOTE("(RESOLVER) Service '%s' of type '%s' in domain "
                              "'%s' is regesterd with a different address.\n",
                              name, type, domain);
      } else {
        POCL_MSG_PRINT_REMOTE("(RESOLVER) Service '%s' of type '%s' in domain "
                              "'%s' is being added.\n",
                              name, type, domain);
        register_server(S, name, type, domain, txt, key);
      }
    }
  }
  }
  avahi_service_resolver_free(r);
}

static void browse_callback(AvahiServiceBrowser *b, AvahiIfIndex interface,
                            AvahiProtocol protocol, AvahiBrowserEvent event,
                            const char *name, const char *type,
                            const char *domain,
                            AVAHI_GCC_UNUSED AvahiLookupResultFlags flags,
                            void *userdata) {
  assert(b);
  /* Called whenever a new services becomes available on the LAN or is removed
   * from the LAN */
  switch (event) {
  case AVAHI_BROWSER_FAILURE:
    POCL_MSG_ERR("(Browser) %s\n", avahi_strerror(avahi_client_errno(
                                       avahi_service_browser_get_client(b))));
    return;

  case AVAHI_BROWSER_NEW:
    POCL_MSG_PRINT_REMOTE(
        "(Browser) NEW: service '%s' of type '%s' in domain '%s'\n", name, type,
        domain);
    /* We ignore the returned resolver object. In the callback
       function we free it. If the server is terminated before
       the callback function is called the server will free
       the resolver for us. */
    if (!(avahi_service_resolver_new(client, interface, protocol, name, type,
                                     domain, AVAHI_PROTO_UNSPEC, 0,
                                     resolve_callback, NULL)))
      POCL_MSG_ERR("Failed to resolve service '%s': %s\n", name,
                   avahi_strerror(avahi_client_errno(client)));
    break;

  case AVAHI_BROWSER_REMOVE:
    POCL_MSG_PRINT_REMOTE(
        "(Browser) REMOVE: service '%s' of type '%s' in domain '%s'\n", name,
        type, domain);
    break;

  case AVAHI_BROWSER_ALL_FOR_NOW:

  case AVAHI_BROWSER_CACHE_EXHAUSTED:
    POCL_MSG_PRINT_REMOTE("(Browser) %s\n",
                          event == AVAHI_BROWSER_CACHE_EXHAUSTED
                              ? "CACHE_EXHAUSTED"
                              : "ALL_FOR_NOW");
    break;
  }
}

static void discovery_client_callback(AvahiClient *c, AvahiClientState state,
                                      AVAHI_GCC_UNUSED void *userdata) {
  assert(c);
  /* Called whenever the client or server state changes */
  if (state == AVAHI_CLIENT_FAILURE) {

    POCL_MSG_ERR("Server connection failure: %s\n",
                 avahi_strerror(avahi_client_errno(c)));
    avahi_threaded_poll_quit(thread_poll);
  }
}

void stop_mDNS() {
  if (client)
    avahi_client_free(client);
  if (sb) {
    int i = 0;
    while (i < domain_count) {
      avahi_service_browser_free(sb[i]);
      i++;
    }
    free(sb);
    DestroyServiceInfoTable(ServiceInfoTable);
  }
  if (thread_poll) {
    avahi_threaded_poll_quit(thread_poll);
    avahi_threaded_poll_free(thread_poll);
    thread_poll = NULL;
  }
}

int mDNS_SD(cl_int (*discovery_callback)(char *const, unsigned),
            unsigned dev_type,
            cl_int (*pocl_remote_discovered_server_reconnect)(const char *)) {

  discovery_cb = discovery_callback;
  dev_typ = dev_type;
  pocl_remote_reconnect = pocl_remote_discovered_server_reconnect;
  int errcode = CL_SUCCESS;

  /* Allocate main loop object */
  POCL_GOTO_ERROR_ON(!(thread_poll = avahi_threaded_poll_new()),
                     CL_DEVICE_NOT_AVAILABLE,
                     "Failed to create simple poll object.\n");

  /* Allocate a new client */
  POCL_GOTO_ERROR_ON(
      !(client = avahi_client_new(avahi_threaded_poll_get(thread_poll), 0,
                                  discovery_client_callback, NULL, &errcode)),
      CL_DEVICE_NOT_AVAILABLE, "Failed to create client: %s\n",
      avahi_strerror(errcode));

  const char *env = pocl_get_string_option(POCL_REMOTE_SEARCH_DOMAINS, NULL);
  if (env && *env) {
    // First count number of domains
    char *domains;
    char *token;
    // at least one domain present as env is not NULL and one for .local
    domain_count = 1 + 1;
    domains = strdup(env);
    token = strtok(domains, ",");
    while (token) {
      token = strtok(NULL, ",");
      domain_count++;
    }
    free(domains);
    domains = strdup(env);
    int i = 1;
    sb = (AvahiServiceBrowser **)malloc(domain_count *
                                        sizeof(AvahiServiceBrowser *));

    // service browser for .local (mDNS)
    POCL_GOTO_ERROR_ON(!(sb[0] = avahi_service_browser_new(
                             client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
                             "_pocl._tcp", NULL, 0, browse_callback, NULL)),
                       CL_DEVICE_NOT_AVAILABLE,
                       "Failed to create service browser: %s\n",
                       avahi_strerror(avahi_client_errno(client)));

    token = strtok(domains, ",");

    while (token) {
      // service browser for domain currently in token variable
      POCL_GOTO_ERROR_ON(!(sb[i] = avahi_service_browser_new(
                               client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
                               "_pocl._tcp", token, 0, browse_callback, NULL)),
                         CL_DEVICE_NOT_AVAILABLE,
                         "Failed to create service browser: %s\n",
                         avahi_strerror(avahi_client_errno(client)));
      i++;
      token = strtok(NULL, ",");
    }
    free(domains);
  } else {
  
    sb = (AvahiServiceBrowser **)malloc(sizeof(AvahiServiceBrowser *));
    domain_count = 1;
    // service browser for .local (mDNS)
    POCL_GOTO_ERROR_ON(!(sb[0] = avahi_service_browser_new(
                             client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
                             "_pocl._tcp", NULL, 0, browse_callback, NULL)),
                       CL_DEVICE_NOT_AVAILABLE,
                       "Failed to create service browser: %s\n",
                       avahi_strerror(avahi_client_errno(client)));
  }

  /* Run the main loop */
  POCL_GOTO_ERROR_ON((avahi_threaded_poll_start(thread_poll) == -1),
                     CL_DEVICE_NOT_AVAILABLE, "Failed to start avahi poll.\n");
  POCL_MSG_PRINT_REMOTE("(Browser): Browsing started \n");
  return errcode;

ERROR:
  /* clean up*/
  if (client)
    avahi_client_free(client);

  if (sb) {
    int i = 0;
    while (i < domain_count) {
      avahi_service_browser_free(sb[i]);
      i++;
    }
    free(sb);
  }

  if (thread_poll)
    avahi_threaded_poll_free(thread_poll);

  return errcode;
}