#include "contiki.h"
#include "net/ipv6/simple-udp.h"
#include "net/routing/routing.h"
#include "sys/log.h"
#include "sys/clock.h"
#include "net/packetbuf.h"
#include <stdint.h>
#include <string.h>

#define LOG_MODULE "UDP-Server"
#define LOG_LEVEL LOG_LEVEL_INFO

#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678

#define SUMMARY_INTERVAL   (60 * CLOCK_SECOND)
#define FINAL_LOG_TIME     (3590 * CLOCK_SECOND)

/* batas maksimum delay yang masih dianggap valid */
#define MAX_VALID_DELAY    (60 * CLOCK_SECOND)

static struct simple_udp_connection udp_conn;

typedef struct {
  uint32_t seq_id;
  uint32_t timestamp;
} payload_t;

/* global counters (tetap ringan) */
static uint32_t last_seq_id = 0;
static uint32_t total_received = 0;
static uint32_t total_lost = 0;

/* delay stats: hanya untuk paket dengan delay valid */
static uint32_t valid_delay_count = 0;
static uint32_t invalid_delay_count = 0;
static uint32_t delay_sum_ms = 0;
static uint32_t delay_max_ms = 0;

/* RSSI stats */
static int32_t rssi_sum = 0;
static uint32_t rssi_count = 0;

/*---------------------------------------------------------------------------*/
PROCESS(udp_server_process, "UDP Server Clean Final");
AUTOSTART_PROCESSES(&udp_server_process);
/*---------------------------------------------------------------------------*/
static void
udp_rx_callback(struct simple_udp_connection *c,
                const uip_ipaddr_t *sender_addr,
                uint16_t sender_port,
                const uip_ipaddr_t *receiver_addr,
                uint16_t receiver_port,
                const uint8_t *data,
                uint16_t datalen)
{
  (void)c;
  (void)sender_port;
  (void)receiver_addr;
  (void)receiver_port;

  if(datalen == sizeof(payload_t)) {
    payload_t pkt;
    memcpy(&pkt, data, sizeof(pkt));

    clock_time_t now = clock_time();
    uint32_t delay_ms = 0;
    uint8_t delay_valid = 0;

    /* =========================
     * Delay sanity check
     * ========================= */
    if(now >= pkt.timestamp) {
      clock_time_t delta = now - pkt.timestamp;

      if(delta <= MAX_VALID_DELAY) {
        delay_ms = (1000UL * (uint32_t)delta) / CLOCK_SECOND;
        delay_valid = 1;

        valid_delay_count++;
        delay_sum_ms += delay_ms;

        if(delay_ms > delay_max_ms) {
          delay_max_ms = delay_ms;
        }
      } else {
        invalid_delay_count++;
      }
    } else {
      /* timestamp anomaly */
      invalid_delay_count++;
    }

    int16_t rssi = packetbuf_attr(PACKETBUF_ATTR_RSSI);
    rssi_sum += rssi;
    rssi_count++;

    /* =========================
     * Sequence / loss tracking
     * CATATAN:
     * ini masih model global tunggal,
     * jadi valid penuh hanya jika satu sender.
     * untuk multi-client, ini tetap aproksimasi.
     * ========================= */
    if(total_received == 0) {
      last_seq_id = pkt.seq_id;
    } else if(pkt.seq_id > last_seq_id + 1) {
      total_lost += (pkt.seq_id - last_seq_id - 1);
    }

    last_seq_id = pkt.seq_id;
    total_received++;

    {
      uint32_t total = total_received + total_lost;
      uint32_t pdr_percent = (total > 0) ? (100UL * total_received) / total : 0;

      if(delay_valid) {
        LOG_INFO("📥 From ");
        LOG_INFO_6ADDR(sender_addr);
        LOG_INFO_(", Seq=%lu Delay=%lums RSSI=%ddBm Recv=%lu Lost=%lu PDR=%lu%%\n",
                  (unsigned long)pkt.seq_id,
                  (unsigned long)delay_ms,
                  (int)rssi,
                  (unsigned long)total_received,
                  (unsigned long)total_lost,
                  (unsigned long)pdr_percent);
      } else {
        LOG_INFO("📥 From ");
        LOG_INFO_6ADDR(sender_addr);
        LOG_INFO_(", Seq=%lu RSSI=%ddBm Recv=%lu Lost=%lu PDR=%lu%%\n",
                  (unsigned long)pkt.seq_id,
                  (int)rssi,
                  (unsigned long)total_received,
                  (unsigned long)total_lost,
                  (unsigned long)pdr_percent);
      }
    }

  } else {
    LOG_WARN("⚠️ Invalid payload size: %u\n", datalen);
  }
}
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(udp_server_process, ev, data)
{
  static struct etimer summary_timer;
  static struct etimer final_log_timer;

  PROCESS_BEGIN();

  LOG_INFO("🔧 Setting as root...\n");
  NETSTACK_ROUTING.root_start();

  simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL,
                      UDP_CLIENT_PORT, udp_rx_callback);

  LOG_INFO("🚀 Server listening on port %d\n", UDP_SERVER_PORT);

  etimer_set(&summary_timer, SUMMARY_INTERVAL);
  etimer_set(&final_log_timer, FINAL_LOG_TIME);

  while(1) {
    PROCESS_YIELD();

    if(etimer_expired(&summary_timer)) {
      uint32_t total = total_received + total_lost;
      uint32_t pdr_percent = (total > 0) ? (100UL * total_received) / total : 0;
      uint32_t avg_delay_ms = (valid_delay_count > 0) ?
                              (delay_sum_ms / valid_delay_count) : 0;
      int32_t avg_rssi = (rssi_count > 0) ?
                         (rssi_sum / (int32_t)rssi_count) : 0;

      LOG_INFO("📊 UDP_SUMMARY Recv=%lu Lost=%lu Total=%lu PDR=%lu%% "
               "DelayAvg=%lums DelayMax=%lums RSSIAvg=%ddBm\n",
               (unsigned long)total_received,
               (unsigned long)total_lost,
               (unsigned long)total,
               (unsigned long)pdr_percent,
               (unsigned long)avg_delay_ms,
               (unsigned long)delay_max_ms,
               (int)avg_rssi);

      etimer_reset(&summary_timer);
    }

    if(etimer_expired(&final_log_timer)) {
      uint32_t total = total_received + total_lost;
      uint32_t final_pdr = (total > 0) ? (100UL * total_received) / total : 0;
      uint32_t final_avg_delay = (valid_delay_count > 0) ?
                                 (delay_sum_ms / valid_delay_count) : 0;
      int32_t final_avg_rssi = (rssi_count > 0) ?
                               (rssi_sum / (int32_t)rssi_count) : 0;

      LOG_INFO("✅ FINAL_STATS Received=%lu Lost=%lu Total=%lu PDR=%lu%% "
               "DelayAvg=%lums DelayMax=%lums RSSIAvg=%ddBm\n",
               (unsigned long)total_received,
               (unsigned long)total_lost,
               (unsigned long)total,
               (unsigned long)final_pdr,
               (unsigned long)final_avg_delay,
               (unsigned long)delay_max_ms,
               (int)final_avg_rssi);
      break;
    }
  }

  PROCESS_END();
}



