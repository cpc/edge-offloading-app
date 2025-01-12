#include <avahi-common/address.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <avahi-client/client.h>
#include <avahi-client/lookup.h>
#include <avahi-common/timeval.h>
#include <avahi-common/thread-watch.h>
#include <avahi-common/malloc.h>
#include <avahi-common/error.h>
#include <avahi-client/publish.h>
#include <avahi-common/alternative.h>

#include "create_service.h"
#include "pocl_debug.h"

#define IPV6
#define IPV4

#define MAX_STR_LEN 256

typedef struct ServiceInfo
{
    char *p_service_name;
    int16_t if_index;
    int16_t ip_proto;
    uint16_t listen_port;
    const char *TXT;

}ServiceInfo;

ServiceInfo user_data;

/*xxxx----------------------- CREATE LOCAL SERVICE ------------------------xxxx*/
static AvahiThreadedPoll *poll_discovery = NULL;
static AvahiClient *client_discovery = NULL;
static AvahiEntryGroup *group_discovery = NULL;

static void create_services(AvahiClient *c);

static void entry_group_callback(AvahiEntryGroup *g, AvahiEntryGroupState state, AVAHI_GCC_UNUSED void *userdata)
{
    /* Called whenever the entry group state changes */
    assert(g == group_discovery || group_discovery == NULL);
    group_discovery = g;

    switch (state)
    {
        case AVAHI_ENTRY_GROUP_ESTABLISHED:
            /* The entry group has been established successfully */
            POCL_MSG_PRINT_REMOTE("Service '%s' successfully established.\n", user_data.p_service_name);
            break;

        case AVAHI_ENTRY_GROUP_COLLISION:
        {
            char *n;
            /* A service name collision with a remote service happened. Let's pick a new name */
            n = avahi_alternative_service_name(user_data.p_service_name);
            avahi_free(user_data.p_service_name);
            user_data.p_service_name = strndup( n, MAX_STR_LEN);
            POCL_MSG_PRINT_REMOTE("Service name collision, renaming service to '%s'\n", user_data.p_service_name);
            /* And recreate the services */
            create_services(avahi_entry_group_get_client(g));
            break;
        }
            
        case AVAHI_ENTRY_GROUP_FAILURE:
            POCL_MSG_ERR("Entry group failure: %s\n", avahi_strerror(avahi_client_errno(avahi_entry_group_get_client(g))));
            avahi_threaded_poll_quit(poll_discovery);
            poll_discovery = NULL;
            break;
        case AVAHI_ENTRY_GROUP_UNCOMMITED:
        case AVAHI_ENTRY_GROUP_REGISTERING:
            ;
    }
}

static void create_services(AvahiClient *c)
{
    char *n;
    int ret;
    assert(c);

    /* If this is the first time we're called, let's create a new entry group if necessary */
    if(!group_discovery)
    {
        if(!(group_discovery = avahi_entry_group_new(c, entry_group_callback, NULL)))
        {
            POCL_MSG_ERR("avahi_entry_group_new() failed: %s\n", avahi_strerror(avahi_client_errno(c)));
            ret = -1;
            goto fail;
        }   
    }

    /* If the group is empty (either because it was just created, or because it was reset previously, add our entries.  */
    if(avahi_entry_group_is_empty(group_discovery))
    {
        POCL_MSG_PRINT_REMOTE("Adding service '%s'\n", user_data.p_service_name);
        
        /* We will now add two services and one subtype to the entry group.*/
        if((ret = avahi_entry_group_add_service(group_discovery, user_data.if_index, user_data.ip_proto, 0, user_data.p_service_name, "_pocl._tcp", NULL, NULL, user_data.listen_port, user_data.TXT, NULL)) < 0)
        {
            if (ret == AVAHI_ERR_COLLISION) {
                /* A service name collision with a local service happened. Let's
                * pick a new name */
                n = avahi_alternative_service_name(user_data.p_service_name);
                avahi_free(user_data.p_service_name);
                user_data.p_service_name = strndup( n, MAX_STR_LEN);
                POCL_MSG_PRINT_REMOTE("Service name collision, renaming service to '%s'\n", user_data.p_service_name);
                avahi_entry_group_reset(group_discovery);
                create_services(c);
            }
            POCL_MSG_ERR("Failed to add _pocl._tcp service: %s\n", avahi_strerror(ret));
            goto fail;
        }

        /* Tell the server to register the service */
        if ((ret = avahi_entry_group_commit(group_discovery)) < 0) {
            POCL_MSG_ERR("Failed to commit entry group: %s\n", avahi_strerror(ret));
            goto fail;
        }
    }

    return;
fail:
    avahi_threaded_poll_quit(poll_discovery);
    poll_discovery = NULL;
    return;
}

static void service_client_callback(AvahiClient *c, AvahiClientState state, AVAHI_GCC_UNUSED void* userdata)
{
    /* Called whenever the client or server state changes */
    
    assert(c);
    switch (state)
    {
        case AVAHI_CLIENT_S_RUNNING:
            /* The server has startup successfully and registered its host
             * name on the network, so it's time to create our services */
            create_services(c);
            break;
        
        case AVAHI_CLIENT_FAILURE:
            POCL_MSG_ERR("Client failure: %s\n", avahi_strerror(avahi_client_errno(c)));
            avahi_threaded_poll_quit(poll_discovery);
            break;
        case AVAHI_CLIENT_S_COLLISION:
            /* Let's drop our registered services. When the server is back
             * in AVAHI_SERVER_RUNNING state we will register them
             * again with the new host name. */
        case AVAHI_CLIENT_S_REGISTERING:
            /* The server records are now being established. This
             * might be caused by a host name change. We need to wait
             * for our own records to register until the host name is
             * properly esatblished. */
            if (group_discovery)
                avahi_entry_group_reset(group_discovery);
            break;
        case AVAHI_CLIENT_CONNECTING:
            ;
    }
}

int new_local_service(const char * const p_service_name, int16_t if_index, int16_t ip_proto, uint16_t listen_port, const char * const TXT)
{
    int error;
    int ret = 1;
    struct timeval tv;

    user_data.p_service_name = strndup( p_service_name, MAX_STR_LEN); // We make a copy becasue name may change due to collision
    user_data.if_index = if_index;
    user_data.ip_proto = ip_proto;
    user_data.listen_port = listen_port;
    user_data.TXT = TXT;

    /*Allocate main loop object*/
    if(!(poll_discovery = avahi_threaded_poll_new()))
    {
        POCL_MSG_ERR("Failed to create simple poll object.\n");
        ret = -1;
        goto fail;
    }

    //service_name = avahi_strdup(p_service_name);
    /* Allocate new client*/
    client_discovery = avahi_client_new(avahi_threaded_poll_get(poll_discovery), 0, service_client_callback, NULL, &error);
    /* Check if new client creation succeeded*/
    if(!client_discovery)
    {
        POCL_MSG_ERR("Failed to create simple poll object.\n");
        ret=-2;
        goto fail;
    }

    /* Run the main loop*/
    avahi_threaded_poll_start(poll_discovery);
    ret = 0;

    if(poll_discovery == NULL) {
        avahi_client_free(client_discovery);
        avahi_free(user_data.p_service_name);
    }
    return ret;
fail:

    /* clean up*/
    if(client_discovery)
        avahi_client_free(client_discovery);
    if(poll_discovery)
        avahi_threaded_poll_free(poll_discovery);
    
    poll_discovery = NULL;
    avahi_free(user_data.p_service_name);
    return ret;
}