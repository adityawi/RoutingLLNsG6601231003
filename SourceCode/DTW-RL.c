/*
 * rpl-mrhof.c  (DROP-IN, CLEAN, LOW-RAM)
 * ============================================================
 * BASELINE: MMDEW (Min-Max + SD) objective function for RPL classic.
 * ADD-ON : DTW-RL (Q-learning) for adaptive tau (τ) ONLY.
 *          - Parent selection tetap MMDEW (rank-based).
 *          - RL state  = observasional (continuous) ala Alilou: invTrust & dE (EWMA).
 *          - RL action = Δτ (DEC / INC) dengan Δτ proporsional instability.
 *          - HOLD bersifat implisit melalui deadband controller
 *            saat instability < I_DEADBAND.

 *          - τ_target dihitung dari aksi + step, lalu smoothing & clamp [1..14] (init=6).
 *          - Reward    = penalty linear sederhana (negatif) + evidence gating.
 *
 * LOGS   : fokus τ-control (TC) + HO + energy (EN) (minim log biar Z1 aman).
 *
 * Install: replace
 *   os/net/routing/rpl-classic/rpl-mrhof.c
 *
 * NOTE (hook CSMA):
 *   Pastikan csma.c memanggil:
 *     mmdew_rl_on_tx_result(nhop_lladdr, ok);
 *   File ini menyediakan simbol tsb.
 */

#include "contiki.h"

#include "net/routing/rpl-classic/rpl.h"
#include "net/routing/rpl-classic/rpl-private.h"
#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "net/linkaddr.h"

#include "sys/log.h"
#include "sys/clock.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "sys/rtimer.h"
#include "lib/random.h"

#include <math.h>
#include <stdlib.h>   /* labs */
#include <string.h>
#include <stdint.h>

#define LOG_MODULE "RPL-MMDEW"
#define LOG_LEVEL LOG_LEVEL_INFO

/* ======================= MMDEW power model ======================= */
#define CURRENT_CPU_mA 1.8f
#define TX_mA          17.4f
#define RX_mA          18.8f
#define VOLTAGE_V      3.0f

/* ======================= MMDEW history =========================== */
#ifndef MAX_HISTORY
#define MAX_HISTORY 6
#endif

#ifndef MMDEW_MAX_PARENTS
#define MMDEW_MAX_PARENTS 2
#endif

/* ======================= Parent switch hysteresis ======================= */
/* Parent baru harus lebih baik sekian rank dari parent saat ini
 * agar perpindahan tidak terlalu sensitif terhadap fluktuasi kecil.
 */
#ifndef PARENT_SWITCH_MARGIN
#define PARENT_SWITCH_MARGIN 10
#endif


/* ===== fixed-point helpers (centi/milli) ===== */
#define F_CENTI(x)   ((long)((x) * 100.0f))
#define F_MILLI(x)   ((long)((x) * 1000.0f))
#define I_PART_C(xc) ((long)((xc) / 100))
#define F_PART_C(xc) ((long)labs((xc) % 100))
#define I_PART_M(xm) ((long)((xm) / 1000))
#define F_PART_M(xm) ((long)labs((xm) % 1000))

/* ===== helpers ===== */
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float clipf(float x, float lo, float hi) { return maxf(lo, minf(x, hi)); }
static inline float ewma(float prev, float x, float alpha) { return (1.f - alpha) * prev + alpha * x; }

/* ======================= RL (Q-Δtau) config ======================= */
#ifndef RL_TAU_ENABLE
#define RL_TAU_ENABLE 1
#endif

/* τ bounds (seconds) */
static const float TAU_MIN = 1.0f;
static const float TAU_MAX = 14.0f;
static const float TAU0    = 6.0f;
static const float I_DEADBAND = 0.10f;  /* 0.05..0.15, tune */


/* RL action set: explicit Δτ actions.
 * HOLD tidak dijadikan aksi eksplisit;
 * HOLD muncul implisit saat instability < I_DEADBAND.
 */
typedef enum {
  ACT_DEC = 0,  /* decrease τ (more reactive) */
  ACT_INC = 1   /* increase τ (more stable) */
} rl_act_t;

#define ACT_N 2


/* Linear Q: Q(x,a)=w_a^T x, x=[1, invT, dE] */
#define FEAT_N 3

/* Q-learning hyperparams */
static const float ETA_Q    = 0.30f;
static const float GAMMA_Q  = 0.80f;

/* exploration */
static float EPS             = 0.06f;
static const float EPS_DECAY = 0.996f;
static const float EPS_MIN   = 0.010f;

/* evidence gate */
static const uint16_t EVID_MIN = 2;

/* EWMA observation smoothing */
static const float OBS_EW_ALPHA = 0.30f;

/* τ smoothing (controller) */
static const float TAU_SMOOTH = 0.20f;

/* instability / penalty weights (trust dominates) */
static const float W_T = 0.70f;
static const float W_E = 0.30f;

/* step gain for Δτ = K_STEP * instability  (seconds) */
static const float K_STEP = 8.0f;  /* tune: 4..8 typical; keep 6 default */
/* reward shaping: encourage larger tau when penalty is low (stable) */
static const float LAMBDA_TAU = 0.30f; /* tuning: 0.15..0.50 */



/* EWMA observasi (global) */
static float ew_invT = 0.0f; /* [0..1] */
static float ew_dE   = 0.0f; /* [0..1] */

/* window counters (event-driven evidence) */
static uint16_t win_ok = 0, win_fail = 0;
static uint16_t win_ho = 0;   /* HO events in current RL window */
static uint32_t win_idx = 0;




/* τ current */
static float tau_cur = 6.0f;

/* Linear weights: Wq[a][k] */
static float Wq[ACT_N][FEAT_N];
static uint8_t q_inited = 0;

/* last (x,a) */
static float x_last[FEAT_N];
static int   a_last = ACT_INC;

/* energy snapshot for dE */
static float e_cpu0, e_tx0, e_rx0;
static uint64_t ec_cpu0_mmdew = 0;   /* khusus validasi EC ala MMDEW */


/* ======================= HO (baseline logging) =================== */
static uint8_t  handover_counter = 0;
static clock_time_t last_ho_time = 0;
static clock_time_t convergence_start_time = 0;
static clock_time_t convergence_end_time   = 0;
static uint8_t convergence_logged = 0;
static linkaddr_t last_selected_lladdr_cache;
static uint8_t last_lladdr_initialized = 0;

/* ======================= per-parent history ====================== */
typedef struct {
  uint8_t    in_use;
  linkaddr_t addr;

  float etx_history[MAX_HISTORY];
  float energy_history[MAX_HISTORY];
  float snr_history[MAX_HISTORY];
  float ho_history[MAX_HISTORY];

  int   history_index;
  uint8_t history_updated;

  uint16_t ho_local_counter;
} mmdew_parent_hist_t;

static mmdew_parent_hist_t ph[MMDEW_MAX_PARENTS];

/* ---------- external (provided by Contiki-NG) ---------- */
const linkaddr_t *rpl_get_parent_lladdr(rpl_parent_t *p);

/* ======================= MMDEW helpers =========================== */
static float get_moving_min(const float *array)
{
  float m = array[0];
  for(int i = 1; i < MAX_HISTORY; i++) {
    if(array[i] < m) m = array[i];
  }
  return m;
}

static float get_moving_max(const float *array)
{
  float M = array[0];
  for(int i = 1; i < MAX_HISTORY; i++) {
    if(array[i] > M) M = array[i];
  }
  return M;
}

static float compute_stddev(const float *array, float minv, float maxv)
{
  float sum = 0.f, sum_sq = 0.f;
  for(int i = 0; i < MAX_HISTORY; i++) {
    float norm = (array[i] - minv) / (maxv - minv + 0.0001f);
    sum += norm;
    sum_sq += norm * norm;
  }
  float mean = sum / (float)MAX_HISTORY;
  float var  = (sum_sq / (float)MAX_HISTORY) - (mean * mean);
  if(var < 0.f) var = 0.f;
  return sqrtf(var);
}

static mmdew_parent_hist_t *get_parent_hist(const linkaddr_t *a)
{
  int free_i = -1;
  for(int i = 0; i < MMDEW_MAX_PARENTS; i++) {
    if(ph[i].in_use && linkaddr_cmp(&ph[i].addr, a)) {
      return &ph[i];
    }
    if(!ph[i].in_use && free_i < 0) {
      free_i = i;
    }
  }

  int idx = (free_i >= 0) ? free_i : 0;
  mmdew_parent_hist_t *s = &ph[idx];
  memset(s, 0, sizeof(*s));
  s->in_use = 1;
  linkaddr_copy(&s->addr, a);
  s->history_index = 0;
  s->history_updated = 0;
  s->ho_local_counter = 0;

  /* safe defaults */
  for(int i = 0; i < MAX_HISTORY; i++) {
    s->etx_history[i]    = 1.0f;
    s->energy_history[i] = 0.2f;
    s->snr_history[i]    = 10.0f;
    s->ho_history[i]     = 1.0f;
  }
  return s;
}

static void update_parent_history(mmdew_parent_hist_t *s,
                                  float etx, float energy_mJ,
                                  float snr, float ho_local)
{
  int i = s->history_index;
  s->etx_history[i]    = etx;
  s->energy_history[i] = energy_mJ;
  s->snr_history[i]    = snr;
  s->ho_history[i]     = ho_local;

  s->history_index = (i + 1) % MAX_HISTORY;
  s->history_updated = 1;
}

/* ======================= MRHOF hooks ============================= */
static uint16_t parent_link_metric(rpl_parent_t *p)
{
  const struct link_stats *stats = rpl_get_parent_link_stats(p);
  return stats != NULL ? stats->etx : 0xffff;
}

static int parent_has_usable_link(rpl_parent_t *p)
{
  return parent_link_metric(p) <= 1024;
}

static uint16_t parent_path_cost(rpl_parent_t *p)
{
  return p != NULL ? p->rank : 0xffff;
}

/* ======================= RL helpers ============================== */
static inline float rand01(void)
{
  return (float)random_rand() / (float)RANDOM_RAND_MAX;
}

static void rl_energy_snapshot(float *cpu, float *tx, float *rx)
{
  energest_flush();
  *cpu = (float)energest_type_time(ENERGEST_TYPE_CPU);
  *tx  = (float)energest_type_time(ENERGEST_TYPE_TRANSMIT);
  *rx  = (float)energest_type_time(ENERGEST_TYPE_LISTEN);
}

/* ===== snapshot CPU untuk validasi EC ala MMDEW ===== */
static void rl_ec_mmdew_snapshot(void)
{
  energest_flush();
  ec_cpu0_mmdew = energest_type_time(ENERGEST_TYPE_CPU);
}

/* ===== hitung ΔEC ala MMDEW (CPU-only, mJ) ===== */
static float rl_ec_mmdew_delta_mJ(void)
{
  energest_flush();
  uint64_t c1 = energest_type_time(ENERGEST_TYPE_CPU);
  float d_cpu = (float)(c1 - ec_cpu0_mmdew);

  return (d_cpu * CURRENT_CPU_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;
}






/* cheap normalize (ticks delta / 1e6) -> [0..1] */
static float rl_energy_change_norm(void)
{
  float c1, t1, r1;
  rl_energy_snapshot(&c1, &t1, &r1);
  float d = (c1 - e_cpu0) + (t1 - e_tx0) + (r1 - e_rx0);
  return clipf(d / 1e6f, 0.f, 1.f);
}

/* ===== RL energy in mJ (CPU + TX + RX) ===== */
static float rl_energy_change_mJ(void)
{
  float c1, t1, r1;
  rl_energy_snapshot(&c1, &t1, &r1);

  float d_cpu = c1 - e_cpu0;
  float d_tx  = t1 - e_tx0;
  float d_rx  = r1 - e_rx0;

  float e_cpu =
    (d_cpu * CURRENT_CPU_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;

  float e_tx =
    (d_tx * TX_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;

  float e_rx =
    (d_rx * RX_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;

  return e_cpu + e_tx + e_rx;
}




static void rl_q_init_once(void)
{
  if(q_inited) return;
  for(int a = 0; a < ACT_N; a++) {
    for(int k = 0; k < FEAT_N; k++) {
      Wq[a][k] = 0.0f;
    }
  }
  q_inited = 1;
}

/* x = [1, invT, dE] */
static void rl_build_x(float invT, float dE, float x[FEAT_N])
{
  x[0] = 1.0f;
  x[1] = clipf(invT, 0.f, 1.f);
  x[2] = clipf(dE,   0.f, 1.f);
}

/* Q(x,a) = dot(Wq[a], x) */
static float rl_Q_of(const float x[FEAT_N], int a)
{
  float q = 0.f;
  for(int k = 0; k < FEAT_N; k++) {
    q += Wq[a][k] * x[k];
  }
  return q;
}

static float rl_maxQx(const float x[FEAT_N])
{
  float m = rl_Q_of(x, 0);
  for(int a = 1; a < ACT_N; a++) {
    float q = rl_Q_of(x, a);
    if(q > m) m = q;
  }
  return m;
}

static int rl_choose_action_x(const float x[FEAT_N])
{
  if(rand01() < EPS) {
    return (int)(random_rand() % ACT_N);
  }

  float q_dec = rl_Q_of(x, ACT_DEC);
  float q_inc = rl_Q_of(x, ACT_INC);

  /* tie-break: bias ke INC agar tidak drift turun tanpa alasan */
  if(fabsf(q_dec - q_inc) < 1e-6f) {
    return ACT_INC;
  }

  return (q_dec > q_inc) ? ACT_DEC : ACT_INC;
}




/* instability (0 good .. 1 bad): combine invT & dE */

static const float W_HO = 0.25f;   /* weight HO influence */

static float rl_instability(float invT, float dE, float ho_win)
{
  float bad =
    (W_T  * clipf(invT,0.f,1.f) +
     W_E  * clipf(dE,0.f,1.f) +
     W_HO * clipf(ho_win,0.f,1.f))
    / (W_T + W_E + W_HO);

  return clipf(bad, 0.f, 1.f);
}


/* Reward (A): penalty linear sederhana (negatif).
 * penalty = weighted(invT, dE) in [0..1]
 * reward  = -penalty           in [-1..0]
 *
 * Stabil untuk pembelajaran online (embedded).
 */
static float rl_reward_penalty_plus_tau(float invT, float dE, float tau_now)
{
  float pen = (W_T * clipf(invT,0.f,1.f) + W_E * clipf(dE,0.f,1.f)) / (W_T + W_E);
  pen = clipf(pen, 0.f, 1.f); /* [0..1] */

  float tau_norm = (tau_now - TAU_MIN) / (TAU_MAX - TAU_MIN);
  tau_norm = clipf(tau_norm, 0.f, 1.f); /* [0..1] */

  /* reward: tetap penalize kondisi buruk, tapi beri bonus tau besar saat kondisi baik */
  float r = -pen + LAMBDA_TAU * (1.f - pen) * tau_norm;

  return clipf(r, -1.f, 1.f);
}


/* Apply RL action to produce tau_target (before smoothing/clamp) */
static float rl_apply_action_to_tau(float tau_prev, int act, float instab)
{
  float step = K_STEP * clipf(instab, 0.f, 1.f);
  float tau_t = (act == ACT_DEC) ? (tau_prev - step) : (tau_prev + step);
  return clipf(tau_t, TAU_MIN, TAU_MAX);
}


/* Smooth + clamp final τ */
static float tau_smooth_clamp(float tau_prev, float tau_target)
{
  float tau_new = (1.f - TAU_SMOOTH) * tau_prev + TAU_SMOOTH * tau_target;
  return clipf(tau_new, TAU_MIN, TAU_MAX);
}

/* ======================= RL API (called by CSMA hook) ============ */
/* Called per TX result (event-driven evidence) */
void mmdew_rl_on_tx_result(const linkaddr_t *nhop, int ok)
{
  (void)nhop; /* tau-RL global */
#if RL_TAU_ENABLE
  if(ok) win_ok++;
  else   win_fail++;
#else
  (void)ok;
#endif
}

/* keep symbol for linkage (optional hook) */
void mmdew_rl_on_possible_loop(const linkaddr_t *curr_parent)
{
  (void)curr_parent;
}

/* ======================= RL process (windowed τ) ================= */
PROCESS(mmdew_rl_proc, "MMDEW tau-RL");

static uint8_t rl_started = 0;
void mmdew_rl_start(void)
{
#if RL_TAU_ENABLE
  if(!rl_started) {
    process_start(&mmdew_rl_proc, NULL);
    rl_started = 1;
  }
#else
  (void)rl_started;
#endif
}

PROCESS_THREAD(mmdew_rl_proc, ev, data)
{
#if RL_TAU_ENABLE
  static struct etimer tau_timer;
#endif
  PROCESS_BEGIN();

#if RL_TAU_ENABLE
  (void)ev; (void)data;

  rl_q_init_once();
  tau_cur = TAU0;

  /* init window snapshot */
  rl_energy_snapshot(&e_cpu0, &e_tx0, &e_rx0);
rl_ec_mmdew_snapshot();
  win_ok = win_fail = 0;

  /* init EWMA obs */
  ew_invT = 0.f;
  ew_dE   = 0.f;

  /* init last feature/action */
  rl_build_x(ew_invT, ew_dE, x_last);
  a_last = ACT_INC;

  while(1) {
    etimer_set(&tau_timer, (clock_time_t)(tau_cur * CLOCK_SECOND));
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&tau_timer));

    /* ---- end-of-window ---- */
    uint16_t evid = (uint16_t)(win_ok + win_fail);

    /* invTrust from evidence window */
    float T_aggr = 0.5f;
    if(evid > 0) {
      T_aggr = (float)win_ok / (float)evid; /* [0..1] */
    }
    float invT_new = clipf(1.f - T_aggr, 0.f, 1.f);

    /* dE from energest window */
  float dE_new = rl_energy_change_norm();
float dE_mJ  = rl_energy_change_mJ();
float dEC_mmdew_mJ = rl_ec_mmdew_delta_mJ();



    /* EWMA obs */
    ew_invT = ewma(ew_invT, invT_new, OBS_EW_ALPHA);
    ew_dE   = ewma(ew_dE,   dE_new,   OBS_EW_ALPHA);

  /* next feature */
float x_next[FEAT_N];
rl_build_x(ew_invT, ew_dE, x_next);

/* HO rate in this window (binary indicator) */
float ho_rate = (win_ho > 0) ? 1.0f : 0.0f;

/* compute instability */
float instab = rl_instability(ew_invT, ew_dE, ho_rate);

    

/* reward A+: penalty + tau-regularizer */
float r = rl_reward_penalty_plus_tau(ew_invT, ew_dE, tau_cur);

/* evidence gating */
if(evid < EVID_MIN) {
  r = 0.f;
}


    /* Q update (semi-gradient) only if evidence enough */
    if(evid >= EVID_MIN) {
      float q_last = rl_Q_of(x_last, a_last);
      float target = r + GAMMA_Q * rl_maxQx(x_next);
      float td = target - q_last;

      for(int k = 0; k < FEAT_N; k++) {
        Wq[a_last][k] += ETA_Q * td * x_last[k];
        Wq[a_last][k] = clipf(Wq[a_last][k], -2.f, 2.f);
      }
    }

  int a_next = rl_choose_action_x(x_next);

float tau_target;
/* HOLD tidak berasal dari aksi RL eksplisit,
 * tetapi dari deadband controller.
 */
if(instab < I_DEADBAND) {
  tau_target = tau_cur; /* implicit HOLD */
} else {
  tau_target = rl_apply_action_to_tau(tau_cur, a_next, instab);
}


float tau_next = tau_smooth_clamp(tau_cur, tau_target);



    /* decay ε */
    if(EPS > EPS_MIN) EPS *= EPS_DECAY;
    if(EPS < EPS_MIN) EPS = EPS_MIN;

    /* minimal log: τ, τ_target, action, invT, dE, instab, evid */
    long tau_c = F_CENTI(tau_next);
    long tgt_c = F_CENTI(tau_target);
    long inv_m = F_MILLI(ew_invT);
    long de_m  = F_MILLI(ew_dE);
    long demj_c = F_CENTI(dE_mJ);
long dec_c = F_CENTI(dEC_mmdew_mJ);
    long ins_m = F_MILLI(instab);
    long eps_m = F_MILLI(EPS);

  LOG_INFO("TC,W=%lu,N=%u,tau=%ld.%02ld,tgt=%ld.%02ld,a=%d,invT=%ld.%03ld,dE=%ld.%03ld,dE_mJ=%ld.%02ld,dEC=%ld.%02ld,I=%ld.%03ld,evid=%u,eps=%ld.%03ld\n",
         (unsigned long)win_idx,
         (unsigned)node_id,
         I_PART_C(tau_c), F_PART_C(tau_c),
         I_PART_C(tgt_c), F_PART_C(tgt_c),
         (int)a_next,
         I_PART_M(inv_m), F_PART_M(inv_m),
         I_PART_M(de_m),  F_PART_M(de_m),
         I_PART_C(demj_c), F_PART_C(demj_c),
         I_PART_C(dec_c), F_PART_C(dec_c),
         I_PART_M(ins_m), F_PART_M(ins_m),
         (unsigned)evid,
         I_PART_M(eps_m), F_PART_M(eps_m));




    /* advance */
    tau_cur = tau_next;
    for(int k = 0; k < FEAT_N; k++) {
      x_last[k] = x_next[k];
    }
    a_last = a_next;

    /* reset window */
    rl_energy_snapshot(&e_cpu0, &e_tx0, &e_rx0);
rl_ec_mmdew_snapshot();
    win_ok = win_fail = win_ho = 0;
    win_idx++;
  }
#endif /* RL_TAU_ENABLE */

  PROCESS_END();
}

/* ======================= MMDEW core ============================== */
static uint64_t last_ticks_cpu_rank   = 0;
static uint64_t last_ticks_cpu_at_ho  = 0;
static uint64_t last_ticks_tx_at_ho   = 0;
static uint64_t last_ticks_rx_at_ho   = 0;

static rpl_rank_t rank_via_parent(rpl_parent_t *p)
{
#if RL_TAU_ENABLE
  if(!rl_started) {
    mmdew_rl_start();
  }
#endif

  if(p == NULL || p->dag == NULL) {
    return RPL_INFINITE_RANK;
  }



  uint16_t min_hoprankinc = p->dag->instance->min_hoprankinc;

  energest_flush();
  const struct link_stats *stats = rpl_get_parent_link_stats(p);

  float etx = stats ? ((float)stats->etx / (float)LINK_STATS_ETX_DIVISOR) : 1.0f;

  uint64_t curr_ticks_cpu = energest_type_time(ENERGEST_TYPE_CPU);
  float d_cpu_ticks = (float)(curr_ticks_cpu - last_ticks_cpu_rank);

  /* energy proxy (mJ) for MMDEW metric history */
  float energy_mJ = (d_cpu_ticks * CURRENT_CPU_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;
  last_ticks_cpu_rank = curr_ticks_cpu;

  int rssi = stats ? (int)stats->rssi : -100;
  float snr = (float)(rssi - (-85));

  const linkaddr_t *lla = rpl_get_parent_lladdr(p);
  if(lla == NULL) {
    static const linkaddr_t zero = {{0}};
    lla = &zero;
  }

  mmdew_parent_hist_t *H = get_parent_hist(lla);
  float ho_local = (float)H->ho_local_counter;
  update_parent_history(H, etx, energy_mJ, snr, ho_local);

  float norm_etx = 0.f, norm_ec = 0.f, norm_snr = 0.f, norm_ho = 0.f;
  float w_etx = 0.f, w_ec = 0.f, w_snr = 0.f, w_ho = 0.f;

  if(H->history_updated) {
    float etx_min = get_moving_min(H->etx_history);
    float etx_max = get_moving_max(H->etx_history);
    float ec_min  = get_moving_min(H->energy_history);
    float ec_max  = get_moving_max(H->energy_history);
    float snr_min = get_moving_min(H->snr_history);
    float snr_max = get_moving_max(H->snr_history);
    float ho_min  = get_moving_min(H->ho_history);
    float ho_max  = get_moving_max(H->ho_history);

    /* avoid div0 */
    if(etx_max == etx_min) etx_max = etx_min + 0.01f;
    if(ec_max  == ec_min)  ec_max  = ec_min  + 0.01f;
    if(snr_max == snr_min) snr_max = snr_min + 0.01f;
    if(ho_max  == ho_min)  ho_max  = ho_min  + 0.01f;

    /* normalize */
    norm_etx = (etx - etx_min) / (etx_max - etx_min + 0.0001f);
    norm_ec  = (energy_mJ - ec_min) / (ec_max - ec_min + 0.0001f);
    norm_ho  = (ho_local - ho_min) / (ho_max - ho_min + 0.0001f);

    /* for SNR: higher is better -> invert to cost */
    norm_snr = 1.0f - ((snr - snr_min) / (snr_max - snr_min + 0.0001f));

    /* SD weights */
    w_etx = compute_stddev(H->etx_history, etx_min, etx_max);
    w_ec  = compute_stddev(H->energy_history, ec_min, ec_max);
    w_snr = compute_stddev(H->snr_history, snr_min, snr_max);
    w_ho  = compute_stddev(H->ho_history, ho_min, ho_max);

    H->history_updated = 0;
  }

  float w_total = w_etx + w_ec + w_snr + w_ho + 0.0001f;
  float composite_cost =
    (w_etx * norm_etx + w_ec * norm_ec + w_snr * norm_snr + w_ho * norm_ho) / w_total;

  uint16_t R = (uint16_t)(composite_cost * (float)min_hoprankinc);

  return MIN(p->rank + R, RPL_INFINITE_RANK);
}

/* ======================= Parent selection (MMDEW) ================= */
static rpl_parent_t *best_parent(rpl_parent_t *p1, rpl_parent_t *p2)
{
  if(p1 == NULL) return p2;
  if(p2 == NULL) return p1;

  rpl_dag_t *dag = p1->dag;

  uint16_t rank1 = parent_path_cost(p1);
  uint16_t rank2 = parent_path_cost(p2);

  /* kandidat terbaik secara rank */
  rpl_parent_t *best = (rank1 < rank2) ? p1 : p2;
  uint16_t rank_best = (rank1 < rank2) ? rank1 : rank2;

  /* default selected = best */
  rpl_parent_t *selected = best;

  if(dag != NULL) {
    rpl_parent_t *current = dag->preferred_parent;

    /* jika sudah punya parent aktif, pakai hysteresis */
    if(current != NULL) {
      uint16_t rank_current = parent_path_cost(current);

      /* hanya pindah jika parent baru jauh lebih baik */
      if(best != current) {
        if((uint16_t)(rank_best + PARENT_SWITCH_MARGIN) < rank_current) {
          selected = best;      /* switch allowed */
        } else {
          selected = current;   /* stay with current parent */
        }
      } else {
        selected = current;
      }
    }

    /* convergence logging */
    if(dag->preferred_parent == NULL && selected != NULL) {
      convergence_start_time = clock_time();
      convergence_logged = 0;
      LOG_INFO("HO,ConvStart t=%lu\n", (unsigned long)convergence_start_time);
    }

    if(dag->preferred_parent != NULL &&
       selected == dag->preferred_parent &&
       !convergence_logged) {
      convergence_end_time = clock_time();
      convergence_logged = 1;
      clock_time_t dur = convergence_end_time - convergence_start_time;
      LOG_INFO("HO,ConvDone d=%lu\n", (unsigned long)dur);
    }

    /* handover logging */
    if(dag->preferred_parent != NULL && selected != dag->preferred_parent) {
      const linkaddr_t *new_lladdr = rpl_get_parent_lladdr(selected);
      if(new_lladdr != NULL &&
         (!last_lladdr_initialized || !linkaddr_cmp(new_lladdr, &last_selected_lladdr_cache))) {

        last_lladdr_initialized = 1;
        linkaddr_copy(&last_selected_lladdr_cache, new_lladdr);

        handover_counter++;
        win_ho++;

        clock_time_t now = clock_time();
        clock_time_t ho_delay = now - last_ho_time;
        last_ho_time = now;

        LOG_INFO("HO,%u,delay_ms=%lu\n",
                 (unsigned)handover_counter,
                 (unsigned long)(1000UL * (unsigned long)ho_delay) / CLOCK_SECOND);

        /* energy since last HO */
        energest_flush();
        uint64_t curr_cpu = energest_type_time(ENERGEST_TYPE_CPU);
        uint64_t curr_tx  = energest_type_time(ENERGEST_TYPE_TRANSMIT);
        uint64_t curr_rx  = energest_type_time(ENERGEST_TYPE_LISTEN);

        float e_cpu = ((float)(curr_cpu - last_ticks_cpu_at_ho) * CURRENT_CPU_mA / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_tx  = ((float)(curr_tx  - last_ticks_tx_at_ho ) * TX_mA          / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_rx  = ((float)(curr_rx  - last_ticks_rx_at_ho ) * RX_mA          / (float)RTIMER_ARCH_SECOND) * VOLTAGE_V;
        float e_total = e_cpu + e_tx + e_rx;

        last_ticks_cpu_at_ho = curr_cpu;
        last_ticks_tx_at_ho  = curr_tx;
        last_ticks_rx_at_ho  = curr_rx;

        long tot_c = F_CENTI(e_total);
        LOG_INFO("EN,HO,%u,total_mJ=%ld.%02ld\n",
                 (unsigned)handover_counter,
                 I_PART_C(tot_c), F_PART_C(tot_c));

        /* fairness: increment local HO count for that parent */
        mmdew_parent_hist_t *Hsel = get_parent_hist(new_lladdr);
        if(Hsel) {
          Hsel->ho_local_counter++;
        }

        /* reset convergence tracking after HO */
        convergence_logged = 0;
        convergence_start_time = now;
      }
    }
  }

  return selected;
}



/* ======================= Remaining OF glue ======================= */
static void reset(rpl_dag_t *dag) { (void)dag; }

static rpl_dag_t *best_dag(rpl_dag_t *d1, rpl_dag_t *d2)
{
  return d1->rank < d2->rank ? d1 : d2;
}

static void update_metric_container(rpl_instance_t *instance)
{
  instance->mc.type = RPL_DAG_MC_NONE;
}

clock_time_t mmdew_get_last_ho_time(void)
{
  return last_ho_time;
}

/* OF instance */
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










