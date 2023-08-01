# Experiment Configurations for ECRec

## Keys for Evaluation

Note: ECRM, an earlier name of our system, refers to ECRec in this document.

- Modes of running the DLRM training system. These are mostly controlled by which branch to run.

  - XDL: original XDL system with no fault tolerance. This is the `master` branch.
  - ECRM-kX: ECRM with value of parameter k being X. This is the version of ECRM described above including 2PC, neural network replication, etc. This is the `ecrec` branch. The parameter `k` is controlled by `PARITY_K` in [`xdl/ps-plus/ps-plus/common/base_parity_utils.h`](xdl/ps-plus/ps-plus/common/base_parity_utils.h). You will need to change `PARITY_N` at the same time.
  - Ckpt-X: XDL with checkpointing, where checkpoints are written every X minutes. This is the `master` branch. The checkpoint frequency is specified in [`xdl/examples/criteo/criteo_training.py`](xdl/examples/criteo/criteo_training.py).
- DLRMs on which to train. These are controlled by the variable settings in [`xdl/examples/criteo/criteo_training.py`](xdl/examples/criteo/criteo_training.py). The x and y in Criteo-xS-yD are controlled by the 4th and 5th argument to function `xdl.embedding` in these files. Note we have other scripts available in [`xdl/examples/criteo`](xdl/examples/criteo).
  - Criteo: original Criteo DLRM
  - Criteo-2S: as described in paper—2x the number of embedding table entries
  - Criteo-2S-2D: as described in paper—2x the number of embedding table entries, and each entry being twice as wide (e.g., 64 dense rather than 32)
    - Important note: see in paper that a difference instance type is used when running this setup
- Batch size is 2048. This is set in [`xdl/examples/criteo/criteo_training.py`](xdl/examples/criteo/criteo_training.py).

## Evaluations

- Normal mode 1
  - Using the same evaluation setup described in the paper
    - For each of [Criteo, Criteo-2S, Criteo-2S-2D, Criteo-4S, Criteo-8S]
      - For each of [XDL, Ckpt-30, Ckpt-60, ECRM-k2, ECRM-k4]
        - Run the DLRM for 2 hours. Keep all of the logs
  - At the end of this we will be able to plot Figures 5, 6, 7, 8, and 9
- Recovery mode 1
  - Using the same evaluation setup described in the paper
    - For each of [Criteo, Criteo-2S, Criteo-2S-2D, Criteo-4S, Criteo-8S]
      - For each of [Ckpt-30, Ckpt-60, ECRM-k2, ECRM-k4]
        - Run the DLRM normally for 15 minutes, trigger a failure, and let each mode run for 60 additional minutes
  - At the end of this we want to be able to plot Figures 10, 11, and 12
- Normal mode 4: effect of NN replication, 2PC, etc.
  - The goal here is to determine how much slower things like NN replication and 2PC make ECRM
  - New key for this section:
    - k=4.
    - ECRM-kX: As described above
    - ECRM-kX-minusNNRep: same as ECRM-kX, but without replicating NN params
      - Another way of saying this is "without optimizer state" because then we can recover the "approximate" NN from one of the workers
    - [Maybe don't do this one] ECRM-kX-minus2PC: same as ECRM-kX, but without using 2PC
    - ECRM-kX-minusNNRep-minus2PC: same as ECRM-kX, but without replicating NN params and without using 2PC
      - I think this is the same as Kaige's original branch
  - For mode in [ECRM-k4, ECRM-k4-minusNNRep, ECRM-k4-minusNNRep-minus2PC (Kaige's branch)]
    - For DLRM in [Criteo-8S, Criteo-4S, Criteo-2S, Criteo, Criteo-2S-2D]
      - Run in normal mode. Save all the logs
  - Here, we will want to compare the average throughput across each mode
- Recovery mode 2: effect of lock granularity
  - Using the same evaluation setup described in the paper
    - For each of [Criteo-8S, Criteo] (later do Criteo-4S, Criteo-2S, Criteo-2S2D)
      - For each mode in [ECRM-k4]
        - For num\_locks in [1, 10]
          - Trigger a failure, keep all the logs
          - Measure how long it took to fully recover
  - At the end of this we want to compare how long it takes to fully recover from a failure with 1 lock vs. with 10 locks
- Normal mode 2: effect of limited resources
  - Which DLRM we choose to run here depends on the results we see from Normal Mode 1
  - For each mode in [ECRM-k4]
    - For each DLRM in [Criteo-8S, Criteo]
      - For each server instance type in [x1e.2xlarge, r5.8xlarge (note that this is different from r5n.8xlarge)]
        - Run DLRM. Keep all of the logs
  - At the end of this, we want to compare the average throughput to that when running on the original instances we considered
- Normal mode 3: effect of increased number of workers
  - For num\_workers in [5, 10, 15, 20, 25]
    - For each DLRM in [Criteo-8S]
      - For each mode in [XDL, ECRM-k4]
        - Run the DLRM in normal mode. Save all the logs
- Normal mode 3.1
  - For num\_workers in [5, 10, 15, 20, 25]
    - For each DLRM in [Criteo-4S, Criteo-2S, Criteo, Criteo-2S2D]
      - For each mode in [XDL, ECRM-k4]
        - Run the DLRM in normal mode. Save all the logs
  - We want to plot something like Figure 9 from this
  - Note that we don't need to rerun checkpointing for this, as the "steady-state" overhead from checkpointing can be computed based on the time it takes to write a checkpoint, the throughput of normal XDL, and the checkpointing frequency.