#ifndef CREATE_SERVICE_H
#define CREATE_SERVICE_H

#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */


int new_local_service(const char * const p_service_name, int16_t if_index, int16_t ip_proto, uint16_t listen_port, const char * const TXT);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif