# FCL-LMSS on DWRL

## Overview


FCL-LMSS experiments on the DWRL dataset.
This repository contains the codebase for Federated Continual Learning (FCL) experiments on the DWRL plastic waste dataset.
The project focuses on vision-based plastic classification under non-IID, streaming data conditions and compares:
(1) a fixed FCL baseline,
(2) a heuristic controller (V4),
and (3) an LLM-guided policy controller (LMSS).

The DWRL dataset is private and not included in this repository.
Only preprocessing logic, training pipelines, and evaluation code are provided.

## Methods Compared
- Fixed FCL baseline
- Heuristic controller (V4)
- LLM-guided controller (LMSS)

## Dataset
- DWRL (private)
- Not included
- Access upon request / internal use only

# Data directory (private)

This project uses the **DWRL plastic waste dataset**, which is private and
not included in this repository.

Expected structure:
```
DWRL_clean/
├── PET/
├── PP/
├── PE/
├── PS/
├── PVC/
├── TETRA/
└── Other/
```

Only preprocessing logic and data loaders are tracked.

## Repository Structure
src/
configs/
training/
...

## Status
Ongoing research (PhD work, TU Graz)

