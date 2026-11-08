# RoutingLLNsG6601231003
Development and evaluation of Reinforcement Learning-based RPL routing schemes for Wireless Sensor Networks (WSNs) in Low-Power and Lossy Networks (LLNs).

# DTW-RL for Contiki-NG

This repository provides the available implementation, simulation materials,
mobility traces, experimental logs, analysis code, and result summaries for the
manuscript:

> **DTW-RL: Adaptive Routing Evaluation Interval Algorithm for Stable and
> Reliable RPL in Mobile IoT Networks**

## Overview

DTW-RL combines MMDEW-based multi-metric preferred-parent selection with a
compact linear Q-learning controller. The controller adjusts when the routing
evaluation is performed according to observed transmission reliability, energy
change, and handover evidence. The current evaluation compares the complete
DTW-RL implementation with MRHOF and MMDEW under matched network, mobility,
radio, traffic, and platform configurations.

## Quick Access to Reproducibility Materials

The materials are separated by function so that the implementation,
experimental inputs, raw observations, and processed results can be inspected
independently.

| Material | Description | Access |
|---|---|---|
| Contiki-NG | Upstream Contiki-NG repository | [Contiki-NG](https://github.com/contiki-ng/contiki-ng) |
| MSP430-gcc | Upstream Contiki-NG repository | [MSP430-gcc](https://drive.google.com/drive/folders/1lJPCmtHKlhq3ShD4c2bTx-ZxizaNWv3s?usp=sharing) |
| Contiki-NG project archive | Original project files used for the reported experiments | [Contiki-NG DTW-RL](https://drive.google.com/drive/folders/1CUULWMXHz0nja5dL8iOZuV9uqTjWGDfP?usp=sharing)  |
| Project configuration | Original `project-conf.h` and related compile-time configuration | [project-conf.h ](https://drive.google.com/file/d/1-Rx8EZkIiGFRKBbaebF3JDS5XP47BZkj/view?usp=sharing) |
| Cooja scenarios | Original `.csc` simulation scenario files | [.csc ](https://drive.google.com/drive/folders/1ok8TeLIQEzLOU9faPBDnCjfYJU5k8QqE?usp=sharing) |
| Mobility traces | Five changing-trajectory realizations for the two- and five-mobile-node configurations | [ Mobility ](https://drive.google.com/drive/folders/1q85_vcWkfLZBBjuBSrDi2XlPAl3UYn46?usp=sharing) |



## Experimental Design

| Item | Configuration |
|---|---|
| Operating system | Contiki-NG |
| Simulator | Cooja |
| Target platform | Zolertia Z1 |
| Network stack | RPL Classic over 6LoWPAN |
| MAC layer | CSMA |
| IEEE 802.15.4 channel | 26 |
| Compared algorithms | MRHOF, MMDEW, and DTW-RL |
| Static sensor nodes | 24 |
| Mobile sensor nodes | 2 or 5 |
| Sink/RPL root | 1 |
| Packet-transmission interval | 10, 20, or 30 seconds |
| Simulation duration | 3,600 seconds per run |
| Mobility realizations | 5 changing trajectories per condition |

The same changing-trajectory realization and experimental configuration should
be applied to all three algorithms within each matched comparison.



## Implementation Files

RPL Classic loads the objective-function implementation from:

```text
os/net/routing/rpl-classic/rpl-mrhof.c
```

For each algorithm build, install the corresponding archived source at that
location and rebuild the project. Keep separate build directories or preserve
the resulting firmware filename so that binaries from different algorithms are
not confused.

The DTW-RL application additionally requires the matching UDP client, UDP
server, `project-conf.h`, and Cooja scenario used for the selected experimental
condition.

## Running Contiki-NG
1. cd contiki-ng/
2. cd tools/
3. cd cooja/
4. export PATH=/opt/mspgcc-4.7.3/bin:$PATH
5. ./gradlew run

## Reproduction Workflow

1. Obtain the Contiki-NG project archive and confirm the recorded version or
   commit information.
2. Select MRHOF, MMDEW, or DTW-RL and install the corresponding objective-
   function source.
3. Apply the supplied `project-conf.h` and related compile-time settings.
4. Open the appropriate Cooja `.csc` scenario.
5. Select the required mobility configuration and changing-trajectory file.
6. Apply the recorded random seed and packet-transmission interval.
7. Run the simulation for 3,600 seconds and export the complete Cooja log as
   CSV with the columns `time`, `node`, and `message`.
8. Run the supplied Python or Colab analysis using the five matched trajectory
   logs for the condition.
9. Retain raw logs, per-run outputs, validation warnings, and processed result
   tables without overwriting the original observations.

Exact commands should be copied from the public run-instruction record linked
in the Quick Access table.

## Logged and Derived Metrics

The available DTW-RL logs support extraction of:

- unique application packets submitted by each non-sink node;
- unique packets received by the sink;
- reconstructed end-to-end packet-delivery ratio;
- packet delay and received-signal strength;
- DTW-RL controller-window updates and interval values;
- DEC and INC controller actions;
- recorded handover events and diagnostic handover delay;
- DIO, DAO, and DIS transmissions;
- local RPL repairs;
- cumulative CPU, radio-transmission, radio-reception, and total energy per
  logged node;
- convergence start/end events; and
- preferred-parent assignment and change events.


## Reproducibility Status

The uploaded project, configuration, scenario, mobility, source-code, and
experimental files should be identified as original only when they are the
actual archived files used in the reported experiments. Reconstructed or newly
created documentation must be labelled separately.

The available log identifies the software build as:

```text
Contiki-NG-develop/v5.0-36-gd8f74d170-dirty
```

The `dirty` suffix indicates that the checked-out source contained local
changes. Therefore, the archived modified files remain necessary even when the
base Git revision is identified. Any unavailable commit metadata, seed record,
or command history should be reported explicitly rather than reconstructed
speculatively.



## License

See [LICENSE](LICENSE).

## Citation

Citation metadata should be provided in [`CITATION.cff`](CITATION.cff). Until
that file is added, cite the associated manuscript and this repository URL.
