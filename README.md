# ⚡ Four-Quadrant Power Analysis & Computational Engine (4PACE)

[![Status](https://img.shields.io/badge/Status-Early%20Access-orange.svg)]()
[![Version](https://img.shields.io/badge/version-v0.3.0a0-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> **⚠️ Early Access Notice (v0.3.0a0)**
> 
> This project has undergone a massive architectural refactoring. It is now a full-fledged Time-Series Optimization Engine capable of both Operational Simulation and Capacity Expansion Planning. 

**4PACE** is a high-precision, commercial-grade Power System Optimization Engine designed for modern electrical grids. Built entirely in Python using cutting-edge convex optimization (CVXPY) and graph theory (NetworkX), 4PACE handles everything from radial microgrids to complex mesh networks like the IEEE 14 Bus System.

---

## 🚀 What's New in v0.3.0a0? (The Architecture Leap)

* **Unified MPOPF Engine:** A powerful Multi-Period Optimal Power Flow engine capable of 24-hour time-series simulations.
* **Switchable Relaxation (SOCP & SDP):** * `relax='SOCP'`: Ultra-fast Branch Flow Model for Radial Distribution Grids.
  * `relax='SDP'`: Highly accurate Bus Injection Model (Hermitian Matrix) for complex Mesh/Ring Networks.
* **Capacity Expansion Planning (CEP):** AI-driven investment decision module. 4PACE can now calculate the optimal tradeoff between CapEx (Installation Costs) and OpEx (Fuel/Operating Costs) to automatically size candidate Batteries (BESS) and Solar Inverters.
* **Robust N-R Validation:** Every optimization plan is automatically validated by a full AC Newton-Raphson solver to ensure 100% physical feasibility (KCL/KVL compliance) and automatic PV-to-PQ switching during reactive power limits.
* **Clean OOP Architecture:** Complete separation of concerns (`model.py` for physics, `psys.py` for topology, `pfa.py` for algorithms).

---

## 🛠️ Quick Start Guide

### 1. Define your grid and candidate assets (`config.yaml`)
Create a configuration file. You can now add `is_candidate: true` and cost parameters to let AI decide the optimal sizing for your grid.

```yaml
Sbase: 100.0
buses:
  - name: "Bus_A"
    Vbase: 115.0
    bus_type: Slack
    components:
      - type: SynchronousMachine
        name: Gen1
        S_rated: 100.0
        a: 100.0
        b: 25.0
        c: 0.005

  - name: "Bus_B"
    Vbase: 115.0
    bus_type: PQ
    components:
      - type: Load
        name: Load_B
        P: 80.0
        Q: 20.0
      # AI will decide how many MW/MWh to build based on CapEx vs OpEx
      - type: Battery
        name: Candidate_BESS
        is_candidate: true
        capex_per_mw: 200000.0
        capex_per_mwh: 350000.0
        max_build_mw: 50.0
        max_build_mwh: 200.0
        lifetime_years: 10
        interest_rate: 0.05
```

### 2. Prepare Time-Series Data (profiles.csv)
Provide 24-hour multipliers for your loads and renewable sources.
```csv
Hour,Load_B,Candidate_PV
0,0.4,0.0
...
12,0.8,1.0
...
19,1.3,0.0
```

### 3. Run the Simulation (Python)
Execute the planning and operation sequence using the highly optimized Core Engine.
```python
import pandas as pd
from fourpace.psys import Grid, CEP
from fourpace.pfa import plan

# Load configurations
grid = Grid.load('config.yaml')
profile_df = pd.read_csv('profiles.csv')

# Step 1: Investment Phase (Capacity Expansion Planning)
# AI calculates optimal sizes for candidate assets (BESS, Solar)
CEP(grid, profile_df, relax='SOCP', solver_name='CLARABEL')

# Step 2: Operational Phase & Physical Validation
# AI dispatches the assets and runs Newton-Raphson validation per step
plan(grid, profile_df, relax='SDP', solver='SCS')
```

---

### 🔬 Under the Hood
4PACE utilizes a "Plan-then-Validate" architecture:

1. Master Plan Generation: Uses Convex Relaxation (SOCP/SDP) to find the global optimum for the entire 24-hour horizon, taking into account inter-temporal constraints (like Battery State of Charge).

2. Physical Check-bill: The results (Voltage and Phase Angles) are fed as a warm-start into the exact non-linear AC Power Flow (Newton-Raphson) to guarantee that the final results strictly obey Kirchhoff's laws.

---

### 📜 License
This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3). See the LICENSE file for details. This ensures that the core mathematical engine remains open and beneficial to the entire engineering community.