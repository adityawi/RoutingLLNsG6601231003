/*
 * MMDEW Objective Function (Min-Max + SD) with:
 * - energest_flush() before reading energy ticks
 * - Per-parent history (by linkaddr_t) for fair normalization
 * - LOG format unchanged
 */

#include "net/routing/rpl-classic/rpl.h"
#include "net/routing/rpl-classic/rpl-private.h"
#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "sys/log.h"
#include "sys/clock.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "net/linkaddr.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_MODULE "RPL-MMDEW"
#define LOG_LEVEL LOG_LEVEL_INFO

#define CURRENT_CPU_mA 1.8
#define TX_mA 17.4f
#define RX_mA 18.8f
#define VOLTAGE_V 3.0

/* jumlah sampel per parent (pakai konstanta lama agar fungsi2 tetap) */
#define MAX_HISTORY 6

/* batasi banyaknya parent yang dilacak demi RAM */
#ifndef MMDEW_MAX_PARENTS
#define MMDEW_MAX_PARENTS 2
#endif

#define ENABLE_R_LOG_FILTER 1
#define R_LOG_THRESHOLD 3

/* ---------- per-parent history buffers ---------- */
typedef struct {
  uint8_t    in_use;
  linkaddr_t addr;

  float etx_history[MAX_HISTORY];
  float energy_history[MAX_HISTORY];
  float snr_history[MAX_HISTORY];
  float ho_history[MAX_HISTORY];
  int   history_index;
  uint8_t history_updated;

  /* ho lokal untuk parent ini (untuk isi ho_history),
   * log tetap pakai handover_counter global agar format sama */
  uint16_t ho_local_counter;
} mmdew_parent_hist_t;

static mmdew_parent_hist_t ph[MMDEW_MAX_PARENTS];

/* ---------- state global lama (tetap) ---------- */
static uint64_t last_ticks_cpu_rank = 0;
static uint64_t last_ticks_cpu_at_ho = 0, last_ticks_tx_at_ho = 0, last_ticks_rx_at_ho = 0;
static uint16_t last_logged_R = 0;
static uint8_t handover_counter = 0;
static clock_time_t last_ho_time = 0;
static clock_time_t convergence_start_time = 0, convergence_end_time = 0;
static uint8_t convergence_logged = 0;
static linkaddr_t last_selected_lladdr_cache;
static uint8_t last_lladdr_initialized = 0;

/* ---------- deklarasi eksternal ---------- */
const linkaddr_t *rpl_get_parent_lladdr(rpl_parent_t *p);

/* ---------- helpers lama (tetap) ---------- */
static float get_moving_min(float *array) {
  float min = array[0];
  for(int i = 1; i < MAX_HISTORY; i++) if(array[i] < min) min = array[i];
  return min;
}
static float get_moving_max(float *array) {
  float max = array[0];
  for(int i = 1; i < MAX_HISTORY; i++) if(array[i] > max) max = array[i];
  return max;
}
static float compute_stddev(float *array, float min, float max) {
  float sum = 0, sum_sq = 0;
  for(int i = 0; i < MAX_HISTORY; i++) {
    float norm = (array[i] - min) / (max - min + 0.0001f);
    sum += norm;
    sum_sq += norm * norm;
  }
  float mean = sum / MAX_HISTORY;
  return sqrtf((sum_sq / MAX_HISTORY) - (mean * mean));
}

/* ---------- per-parent: get / init slot ---------- */
static mmdew_parent_hist_t *get_parent_hist(const linkaddr_t *a)
{
  int free_i = -1;
  for(int i=0;i<MMDEW_MAX_PARENTS;i++){
    if(ph[i].in_use && linkaddr_cmp(&ph[i].addr, a)) return &ph[i];
    if(!ph[i].in_use && free_i<0) free_i = i;
  }
  int idx = (free_i>=0)? free_i : 0; /* reuse index 0 kalau penuh */
  mmdew_parent_hist_t *s = &ph[idx];

  memset(s, 0, sizeof(*s));
  s->in_use = 1;
  linkaddr_copy(&s->addr, a);
  s->history_index = 0;
  s->history_updated = 0;
  s->ho_local_counter = 0;

  /* seed awal (sama seperti default di kode lama) */
  for(int i=0;i<MAX_HISTORY;i++){
    s->etx_history[i]    = 1.0f;
    s->energy_history[i] = 0.2f;
    s->snr_history[i]    = 10.0f;
    s->ho_history[i]     = 1.0f;
  }
  return s;
}

/* ---------- tulis satu sampel ke history parent ---------- */
static void update_parent_history(mmdew_parent_hist_t *s, float etx, float energy_mJ, float snr, float ho_local)
{
  int i = s->history_index;
  s->etx_history[i]    = etx;
  s->energy_history[i] = energy_mJ;
  s->snr_history[i]    = snr;
  s->ho_history[i]     = ho_local;
  s->history_index = (i + 1) % MAX_HISTORY;
  s->history_updated = 1;
}

/* ---------- MRHOF hooks (tetap) ---------- */
static uint16_t parent_link_metric(rpl_parent_t *p) {
  const struct link_stats *stats = rpl_get_parent_link_stats(p);
  return stats != NULL ? stats->etx : 0xffff;
}
static int parent_has_usable_link(rpl_parent_t *p) {
  return parent_link_metric(p) <= 1024;
}

/* ... semua include & define tetap ... */

/* ---------- Inti MMDEW ---------- */
static rpl_rank_t rank_via_parent(rpl_parent_t *p) {
  if(p == NULL || p->dag == NULL) return RPL_INFINITE_RANK;

  uint16_t min_hoprankinc = p->dag->instance->min_hoprankinc;

  /* Penting: flush Energest supaya Δ energi valid */
  energest_flush();

  const struct link_stats *stats = rpl_get_parent_link_stats(p);
  float etx = stats ? (float)stats->etx / LINK_STATS_ETX_DIVISOR : 1.0f;

  uint64_t curr_ticks_cpu = energest_type_time(ENERGEST_TYPE_CPU);
  float energy_mJ = ((float)(curr_ticks_cpu - last_ticks_cpu_rank) * CURRENT_CPU_mA / RTIMER_ARCH_SECOND) * VOLTAGE_V;
  last_ticks_cpu_rank = curr_ticks_cpu;

  int rssi = stats ? stats->rssi : -100;
  float snr = (float)(rssi - (-85)); /* proxy SNR sama seperti kode asli */

  /* per-parent */
  const linkaddr_t *lla = rpl_get_parent_lladdr(p);
  if(lla == NULL) {
    /* fallback aman: pakai slot 0 */
    static const linkaddr_t zero = {{0}};
    lla = &zero;
  }
  mmdew_parent_hist_t *H = get_parent_hist(lla);

  /* HO lokal parent ini (untuk normalisasi ho_history) */
  float ho_local = (float)H->ho_local_counter;

  /* update history per-parent */
  update_parent_history(H, etx, energy_mJ, snr, ho_local);

  /* --- variabel normalisasi & bobot: JANGAN static --- */
  float norm_etx = 0, norm_ec = 0, norm_snr = 0, norm_ho = 0;
  float w_etx = 0, w_ec = 0, w_snr = 0, w_ho = 0;

  if(H->history_updated) {
    float etx_min = get_moving_min(H->etx_history);
    float etx_max = get_moving_max(H->etx_history);
    float ec_min  = get_moving_min(H->energy_history);
    float ec_max  = get_moving_max(H->energy_history);
    float snr_min = get_moving_min(H->snr_history);
    float snr_max = get_moving_max(H->snr_history);
    float ho_min  = get_moving_min(H->ho_history);
    float ho_max  = get_moving_max(H->ho_history);


   if(etx_max == etx_min) etx_max = etx_min + 0.01f;
    if(ec_max  == ec_min)  ec_max  = ec_min  + 0.01f;
    if(snr_max == snr_min) snr_max = snr_min + 0.01f;
    if(ho_max  == ho_min)  ho_max  = ho_min  + 0.01f;

    norm_etx = (etx - etx_min) / (etx_max - etx_min + 0.0001f);
    norm_ec  = (energy_mJ - ec_min) / (ec_max - ec_min + 0.0001f);
    norm_snr = 1.0f - ((snr - snr_min) / (snr_max - snr_min + 0.0001f));
    norm_ho  = (ho_local - ho_min) / (ho_max - ho_min + 0.0001f);

    w_etx = compute_stddev(H->etx_history, etx_min, etx_max);
    w_ec  = compute_stddev(H->energy_history, ec_min, ec_max);
    w_snr = compute_stddev(H->snr_history, snr_min, snr_max);
    w_ho  = compute_stddev(H->ho_history, ho_min, ho_max);

    H->history_updated = 0;
  }

  float w_total = w_etx + w_ec + w_snr + w_ho + 0.0001f;
  float composite_cost = (w_etx * norm_etx + w_ec * norm_ec + w_snr * norm_snr + w_ho * norm_ho) / w_total;
  uint16_t R = (uint16_t)(composite_cost * min_hoprankinc);

#if ENABLE_R_LOG_FILTER
  {
    uint16_t dR = (R > last_logged_R) ? (R - last_logged_R) : (last_logged_R - R);
    if(dR > R_LOG_THRESHOLD) {
      LOG_INFO("MMDEW-SD METRICS: ETX=%u.%02u EC=%lu.%02lumJ SNR=%u.%02udB HO=%u | w=[ETX:%u EC:%u SNR:%u HO:%u] R=%u\n",
        (unsigned)etx, (unsigned)((etx - (int)etx) * 100),
        (unsigned long)energy_mJ, (unsigned long)((energy_mJ - (long)energy_mJ) * 100),
        (unsigned)snr, (unsigned)((snr - (int)snr) * 100),
        handover_counter,
        (unsigned)(w_etx * 100), (unsigned)(w_ec * 100),
        (unsigned)(w_snr * 100), (unsigned)(w_ho * 100),
        R);
      last_logged_R = R;
    }
  }
#endif

  return MIN(p->rank + R, RPL_INFINITE_RANK);
}

static uint16_t parent_path_cost(rpl_parent_t *p) {
  return p != NULL ? p->rank : 0xffff;
}

static rpl_parent_t *best_parent(rpl_parent_t *p1, rpl_parent_t *p2) {
  if(p1 == NULL) return p2;
  if(p2 == NULL) return p1;
  rpl_dag_t *dag = p1->dag;
  uint16_t rank1 = parent_path_cost(p1);
  uint16_t rank2 = parent_path_cost(p2);
  rpl_parent_t *selected = (rank1 < rank2) ? p1 : p2;

  if(dag) {
    if(dag->preferred_parent == NULL && selected != NULL) {
      convergence_start_time = clock_time();
      convergence_logged = 0;
      LOG_INFO("Convergence start at %lu ticks (%lu ms)\n",
               (unsigned long)convergence_start_time,
               (unsigned long)(1000UL * convergence_start_time) / CLOCK_SECOND);
    }
    if(dag->preferred_parent != NULL && selected == dag->preferred_parent && !convergence_logged) {
      convergence_end_time = clock_time();
      convergence_logged = 1;
      clock_time_t convergence_duration = convergence_end_time - convergence_start_time;
      LOG_INFO("Convergence complete at %lu ticks (%lu ms), duration = %lu ticks (%lu ms)\n",
               (unsigned long)convergence_end_time,
               (unsigned long)(1000UL * convergence_end_time) / CLOCK_SECOND,
               (unsigned long)convergence_duration,
               (unsigned long)(1000UL * convergence_duration) / CLOCK_SECOND);
    }
    if(dag->preferred_parent != NULL && selected != dag->preferred_parent) {
      const linkaddr_t *new_lladdr = rpl_get_parent_lladdr(selected);
      if(new_lladdr != NULL && (!last_lladdr_initialized || !linkaddr_cmp(new_lladdr, &last_selected_lladdr_cache))) {
        last_lladdr_initialized = 1;
        linkaddr_copy(&last_selected_lladdr_cache, new_lladdr);

        handover_counter++;
        clock_time_t now = clock_time();
        clock_time_t ho_delay = now - last_ho_time;
        last_ho_time = now;

        LOG_INFO("HO_REAL #%u | Delay = %lu ticks (%lu ms)\n",
                 handover_counter, ho_delay, (1000UL * ho_delay) / CLOCK_SECOND);

        uint64_t curr_cpu = energest_type_time(ENERGEST_TYPE_CPU);
        uint64_t curr_tx = energest_type_time(ENERGEST_TYPE_TRANSMIT);
        uint64_t curr_rx = energest_type_time(ENERGEST_TYPE_LISTEN);

        float e_cpu = ((float)(curr_cpu - last_ticks_cpu_at_ho) * CURRENT_CPU_mA / RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_tx  = ((float)(curr_tx  - last_ticks_tx_at_ho ) * TX_mA       / RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_rx  = ((float)(curr_rx  - last_ticks_rx_at_ho ) * RX_mA       / RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_total = e_cpu + e_tx + e_rx;

        last_ticks_cpu_at_ho = curr_cpu;
        last_ticks_tx_at_ho  = curr_tx;
        last_ticks_rx_at_ho  = curr_rx;

        LOG_INFO("HO_ENERGY #%u | CPU=%lu.%02lumJ TX=%lu.%02lumJ RX=%lu.%02lumJ TOTAL=%lu.%02lumJ\n",
                 handover_counter,
                 (unsigned long)e_cpu, (unsigned long)((e_cpu - (long)e_cpu) * 100),
                 (unsigned long)e_tx,  (unsigned long)((e_tx  - (long)e_tx ) * 100),
                 (unsigned long)e_rx,  (unsigned long)((e_rx  - (long)e_rx ) * 100),
                 (unsigned long)e_total, (unsigned long)((e_total - (long)e_total) * 100));

        LOG_INFO("Parent switch confirmed! HO=%u at %lu ticks (%lu ms), new parent LLADDR=%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n",
                 handover_counter,
                 (unsigned long)last_ho_time,
                 (unsigned long)(1000UL * last_ho_time) / CLOCK_SECOND,
                 new_lladdr->u8[0], new_lladdr->u8[1], new_lladdr->u8[2], new_lladdr->u8[3],
                 new_lladdr->u8[4], new_lladdr->u8[5], new_lladdr->u8[6], new_lladdr->u8[7]);

        /* naikkan HO lokal parent terpilih demi fairness normalisasi HO */
        mmdew_parent_hist_t *Hsel = get_parent_hist(new_lladdr);
        if(Hsel) {
          Hsel->ho_local_counter++;
          /* catatan: pengisian ho_history dilakukan ketika parent ini
             dievaluasi lagi di rank_via_parent() berikutnya */
        }

        convergence_logged = 0;
        convergence_start_time = now;
      }
    }
  }
  return selected;
}

/* ---------- sisanya tanpa perubahan ---------- */
static void reset(rpl_dag_t *dag) {}
static rpl_dag_t *best_dag(rpl_dag_t *d1, rpl_dag_t *d2) { return d1->rank < d2->rank ? d1 : d2; }
static void update_metric_container(rpl_instance_t *instance) { instance->mc.type = RPL_DAG_MC_NONE; }
clock_time_t mmdew_get_last_ho_time(void) { return last_ho_time; }

rpl_of_t rpl_mrhof = {
  reset,
#if RPL_WITH_DAO_ACK
  NULL,
#endif
  parent_link_metric,
  parent_has_usable_link,
  parent_path_cost,
  rank_via_parent,
  best_parent,
  best_dag,
  update_metric_container,
  RPL_OCP_MRHOF
};

