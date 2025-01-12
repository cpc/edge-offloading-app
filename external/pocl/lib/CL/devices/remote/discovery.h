#ifndef DISCOVERY_H
#define DISCOVERY_H

#include <CL/cl.h>

int mDNS_SD(cl_int (* discovery_callback)(char * const, unsigned), unsigned dev_type, cl_int (* pocl_remote_discovered_server_reconnect)(const char *));

#define POCL_REMOTE_SEARCH_DOMAINS "POCL_REMOTE_SEARCH_DOMAINS"
#endif