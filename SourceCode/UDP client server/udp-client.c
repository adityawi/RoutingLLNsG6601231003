#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "sys/log.h"
#include "sys/clock.h"
#include <stdint.h>
#include <string.h>
#include <random.h>
#include <linkaddr.h>

#define LOG_MODULE "UDP-Client"
#define LOG_LEVEL LOG_LEVEL_INFO

#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678
#define BASE_INTERVAL   (20 * CLOCK_SECOND)
#define MAX_JITTER      (1 * CLOCK_SECOND)

static struct simple_udp_connection udp_conn;
static uint32_t tx_count = 0;

typedef struct {
  uint32_t seq_id;
  uint32_t timestamp;
} payload_t;

/*---------------------------------------------------------------------------*/
PROCESS(udp_client_process, "UDP Client No-Reachable-Guard");
AUTOSTART_PROCESSES(&udp_client_process);
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(udp_client_process, ev, data)
{
  static struct etimer periodic_timer;
  uip_ipaddr_t dest_ipaddr;

  PROCESS_BEGIN();

  random_init(linkaddr_node_addr.u8[0]);
  simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL, UDP_SERVER_PORT, NULL);

  LOG_INFO("Client started (without reachable guard)\n");

  while(1) {
    clock_time_t interval = BASE_INTERVAL + (random_rand() % MAX_JITTER);
    etimer_set(&periodic_timer, interval);
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

    /* hanya cek root IP tersedia atau tidak */
    if(NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr)) {
      payload_t pkt;
      pkt.seq_id = tx_count++;
      pkt.timestamp = clock_time();

      simple_udp_sendto(&udp_conn, &pkt, sizeof(pkt), &dest_ipaddr);

      LOG_INFO("Sent Seq=%lu Time=%lu ticks (%lu ms)\n",
               (unsigned long)pkt.seq_id,
               (unsigned long)pkt.timestamp,
               (unsigned long)((1000UL * pkt.timestamp) / CLOCK_SECOND));
    } else {
      LOG_WARN("Root IP unknown, skipped\n");
    }
  }

  PROCESS_END();
}



