# FIREGUARD ID — Project Charter

## 1. Mission & Identity
**Project Name:** FIREGUARD ID — Forest & Land Fire Intelligence, Response, and Resource Optimization System  
**Pilot Area:** Riau Province, Indonesia (`GADM 4.1`, `GID_1: IDN.24_1`)  
**Alignment:** UN SDG 13 (Climate Action), Target 13.1 — strengthen resilience and adaptive capacity to climate-related hazards and natural disasters.

## 2. Core Objectives
1. Build a reproducible, quality-controlled geospatial data pipeline for historical fire detections and environmental context over Riau Province.
2. Characterize spatial fire vulnerability across standardized 0.1° grid cells using **unsupervised spatial clustering** comparing **K-Means** (baseline), **HDBSCAN** (density-based), and **Deep Embedded Clustering (DEC)** (deep learning).
3. Translate cluster profiles and supporting multi-dimensional evidence into defensible **vulnerability assessments** and transparent **spatial prioritization**.
4. Formulate and solve **six operational resource optimization modules** (Base Location, Base Capacity & Infrastructure, Fleet & Equipment, Personnel, Communication & Coordination, Budget & Operational) using MILP/MCLP.
5. Deliver an integrated, config-driven decision-support system with interactive maps, test suite, and complete technical documentation.

## 3. Scope Boundaries
FIREGUARD ID is a **long-term spatial vulnerability characterization and operational resource planning decision-support system**. It is explicitly NOT:
- A 1–3 day supervised hotspot occurrence predictor (that earlier design was superseded).
- An automated fire dispatch system or replacement for emergency authorities (BNPB, BPBD, Manggala Agni).
- A burned-area estimator or unique-fire-incident counter (VIIRS detections are thermal anomaly observations, not unique fire events).
