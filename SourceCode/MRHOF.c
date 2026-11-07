#include "net/routing/rpl-classic/rpl.h"
#include "net/routing/rpl-classic/rpl-private.h"
#include "net/nbr-table.h"
#include "net/link-stats.h"
#include "sys/log.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "net/linkaddr.h"
#include "net/packetbuf.h"

#define LOG_MODULE "RPL"
#define LOG_LEVEL LOG_LEVEL_RPL

#ifdef RPL_MRHOF_CONF_SQUARED_ETX
#define RPL_MRHOF_SQUARED_ETX RPL_MRHOF_CONF_SQUARED_ETX
#else
#define RPL_MRHOF_SQUARED_ETX 0
#endif

#if !RPL_MRHOF_SQUARED_ETX
#define MAX_LINK_METRIC     1024
#define PARENT_SWITCH_THRESHOLD 96
#else
#define MAX_LINK_METRIC     2048
#define PARENT_SWITCH_THRESHOLD 160
#endif

#define MAX_PATH_COST      32768
static unsigned long last_cpu_ticks = 0; // Untuk perhitungan delta CPU
static linkaddr_t last_parent_lladdr = {{0}};
static uint32_t ho_real_count = 0;

static clock_time_t convergence_start_time = 0, convergence_end_time = 0;
static uint8_t convergence_logged = 0;

static clock_time_t ho_convergence_start_time = 0, ho_convergence_end_time = 0;
static uint8_t ho_convergence_logged = 0;

static void reset(rpl_dag_t *dag) {
  LOG_INFO("Reset MRHOF\n");
}

#if RPL_WITH_DAO_ACK
static void dao_ack_callback(rpl_parent_t *p, int status) {
  if(status == RPL_DAO_ACK_UNABLE_TO_ADD_ROUTE_AT_ROOT) return;
  LOG_DBG("MRHOF - DAO ACK received with status: %d\n", status);
  if(status >= RPL_DAO_ACK_UNABLE_TO_ACCEPT || status == RPL_DAO_ACK_TIMEOUT) {
    link_stats_packet_sent(rpl_get_parent_lladdr(p), MAC_TX_OK, 10);
  }
}
#endif

static uint16_t parent_link_metric(rpl_parent_t *p) {
  const struct link_stats *stats = rpl_get_parent_link_stats(p);
  if(stats != NULL) {
#if RPL_MRHOF_SQUARED_ETX
    uint32_t squared_etx = ((uint32_t)stats->etx * stats->etx) / LINK_STATS_ETX_DIVISOR;
    return (uint16_t)MIN(squared_etx, 0xffff);
#else
    return stats->etx;
#endif
  }
  return 0xffff;
}

static uint16_t parent_path_cost(rpl_parent_t *p) {
  uint16_t base;
  if(p == NULL || p->dag == NULL || p->dag->instance == NULL) return 0xffff;

#if RPL_WITH_MC
  switch(p->dag->instance->mc.type) {
    case RPL_DAG_MC_ETX: base = p->mc.obj.etx; break;
    case RPL_DAG_MC_ENERGY: base = p->mc.obj.energy.energy_est << 8; break;
    default: base = p->rank; break;
  }
#else
  base = p->rank;
#endif
  return MIN((uint32_t)base + parent_link_metric(p), 0xffff);
}

// Fungsi untuk menghitung rank berdasarkan parent (Tanpa pengaruh SNR pada rank)
static rpl_rank_t rank_via_parent(rpl_parent_t *p) {
  uint16_t min_hoprankinc, path_cost;
  if(p == NULL || p->dag == NULL || p->dag->instance == NULL) return RPL_INFINITE_RANK;

  min_hoprankinc = p->dag->instance->min_hoprankinc;
  path_cost = parent_path_cost(p);

  rpl_rank_t rank_result = MAX(MIN((uint32_t)p->rank + min_hoprankinc, 0xffff), path_cost);

  // --- Total CPU Energi ---
  uint32_t cpu_ticks_now = energest_type_time(ENERGEST_TYPE_CPU);
  uint32_t cpu_usec = (uint64_t)cpu_ticks_now * 1000000 / RTIMER_SECOND;
  uint32_t ec_cpu_total_uj = (cpu_usec * 3240UL) / 1000;
  uint32_t ec_cpu_total_int = ec_cpu_total_uj / 1000;
  uint32_t ec_cpu_total_frac = (ec_cpu_total_uj % 1000) / 10;

  LOG_INFO("TOTAL_CPU_LOG: Node=%u | CPU_TICKS=%lu | EC_CPU=%lu.%02lu mJ\n",
           node_id, cpu_ticks_now, ec_cpu_total_int, ec_cpu_total_frac);

  // --- Delta CPU Energi (seperti MMDEW) ---
  uint32_t delta_ticks = cpu_ticks_now - last_cpu_ticks;
  last_cpu_ticks = cpu_ticks_now;

  if(delta_ticks > 0) {
    uint32_t delta_usec = (uint64_t)delta_ticks * 1000000 / RTIMER_SECOND;
    uint32_t ec_cpu_delta_uj = (delta_usec * 3240UL) / 1000;
    uint32_t ec_cpu_delta_int = ec_cpu_delta_uj / 1000;
    uint32_t ec_cpu_delta_frac = (ec_cpu_delta_uj % 1000) / 10;

    LOG_INFO("DELTA_CPU_LOG: Node=%u | DELTA_TICKS=%lu | EC_CPU=%lu.%02lu mJ\n",
             node_id, delta_ticks, ec_cpu_delta_int, ec_cpu_delta_frac);
  }



  return rank_result;
}

static int parent_is_acceptable(rpl_parent_t *p) {
  uint16_t link_metric = parent_link_metric(p);
  uint16_t path_cost = parent_path_cost(p);
  return link_metric <= MAX_LINK_METRIC && path_cost <= MAX_PATH_COST;
}

static int parent_has_usable_link(rpl_parent_t *p) {
  uint16_t link_metric = parent_link_metric(p);
  return link_metric <= MAX_LINK_METRIC;
}

rpl_parent_t *best_parent(rpl_parent_t *p1, rpl_parent_t *p2) {
  rpl_dag_t *dag;
  uint16_t p1_cost, p2_cost;
  rpl_parent_t *new_parent = NULL;
  int p1_acceptable = p1 && parent_is_acceptable(p1);
  int p2_acceptable = p2 && parent_is_acceptable(p2);

  if(!p1_acceptable) return p2_acceptable ? p2 : NULL;
  if(!p2_acceptable) return p1_acceptable ? p1 : NULL;

  dag = p1->dag;
  p1_cost = parent_path_cost(p1);
  p2_cost = parent_path_cost(p2);

  if(p1 == dag->preferred_parent || p2 == dag->preferred_parent) {
    if(p1_cost < p2_cost + PARENT_SWITCH_THRESHOLD &&
       p1_cost > p2_cost - PARENT_SWITCH_THRESHOLD) {
      new_parent = dag->preferred_parent;
    }
  }

  if(new_parent == NULL) {
    new_parent = p1_cost < p2_cost ? p1 : p2;
  }

  const linkaddr_t *new_parent_lladdr = rpl_get_parent_lladdr(new_parent);

  // Convergence Time Complete - Log when preferred parent is first set
  if(dag->preferred_parent == NULL && new_parent != NULL) {
    convergence_start_time = clock_time();
    LOG_INFO("Convergence started at %lu ticks\n", convergence_start_time);
  }

  // Convergence Time Complete - Log when preferred parent is stable
  if(dag->preferred_parent != NULL && new_parent == dag->preferred_parent && !convergence_logged) {
    convergence_end_time = clock_time();
    uint32_t duration = convergence_end_time - convergence_start_time;
    LOG_INFO("Convergence complete at %lu ticks, duration: %lu ticks\n", convergence_end_time, duration);
    convergence_logged = 1;
  }

  // Handover Convergence Time
  if(dag->preferred_parent != NULL && new_parent != dag->preferred_parent) {
    // Start HO Convergence
    if(!ho_convergence_logged) {
      ho_convergence_start_time = clock_time();
      LOG_INFO("HO Convergence started at %lu ticks\n", ho_convergence_start_time);
    }
  }

  // End HO Convergence
  if(dag->preferred_parent != NULL && new_parent == dag->preferred_parent && !ho_convergence_logged) {
    ho_convergence_end_time = clock_time();
    uint32_t ho_duration = ho_convergence_end_time - ho_convergence_start_time;
    LOG_INFO("HO Convergence complete at %lu ticks, duration: %lu ticks\n", ho_convergence_end_time, ho_duration);
    ho_convergence_logged = 1;
  }

  // Detect HO real if parent LLADDR changes
  if(!linkaddr_cmp(&last_parent_lladdr, new_parent_lladdr)) {
    ho_real_count++;
    LOG_INFO("HO_REAL_LOG: Node=%u | HO_Count=%lu | From=", node_id, ho_real_count);
    int i;
    for(i = 0; i < LINKADDR_SIZE; i++) {
      if(i > 0) printf(":");
      printf("%02x", last_parent_lladdr.u8[i]);
    }
    printf(" | To=");
    for(i = 0; i < LINKADDR_SIZE; i++) {
      if(i > 0) printf(":");
      printf("%02x", new_parent_lladdr->u8[i]);
    }
    printf("\n");

    // Update last_parent_lladdr
    linkaddr_copy(&last_parent_lladdr, new_parent_lladdr);
  }

  return new_parent;
}


static rpl_dag_t *best_dag(rpl_dag_t *d1, rpl_dag_t *d2) {
  if(d1->grounded != d2->grounded) return d1->grounded ? d1 : d2;
  if(d1->preference != d2->preference) return d1->preference > d2->preference ? d1 : d2;
  return d1->rank < d2->rank ? d1 : d2;
}

#if !RPL_WITH_MC
static void update_metric_container(rpl_instance_t *instance) {
  instance->mc.type = RPL_DAG_MC_NONE;
}
#else
static void update_metric_container(rpl_instance_t *instance) {
  rpl_dag_t *dag = instance->current_dag;
  uint16_t path_cost;
  uint8_t type;

  if(dag == NULL || !dag->joined) {
    LOG_WARN("Cannot update the metric container when not joined\n");
    return;
  }

  if(dag->rank == ROOT_RANK(instance)) {
    instance->mc.type = RPL_DAG_MC;
    instance->mc.flags = 0;
    instance->mc.aggr = RPL_DAG_MC_AGGR_ADDITIVE;
    instance->mc.prec = 0;
    path_cost = dag->rank;
  } else {
    path_cost = parent_path_cost(dag->preferred_parent);
  }

  switch(instance->mc.type) {
    case RPL_DAG_MC_NONE: break;
    case RPL_DAG_MC_ETX:
      instance->mc.length = sizeof(instance->mc.obj.etx);
      instance->mc.obj.etx = path_cost;
      break;
    case RPL_DAG_MC_ENERGY:
      instance->mc.length = sizeof(instance->mc.obj.energy);
      type = dag->rank == ROOT_RANK(instance) ?
             RPL_DAG_MC_ENERGY_TYPE_MAINS : RPL_DAG_MC_ENERGY_TYPE_BATTERY;
      instance->mc.obj.energy.flags = type << RPL_DAG_MC_ENERGY_TYPE;
      instance->mc.obj.energy.energy_est = path_cost >> 8;
      break;
    default:
      LOG_WARN("MRHOF, non-supported MC %u\n", instance->mc.type);
      break;
  }
}
#endif

rpl_of_t rpl_mrhof = {
  reset,
#if RPL_WITH_DAO_ACK
  dao_ack_callback,
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

