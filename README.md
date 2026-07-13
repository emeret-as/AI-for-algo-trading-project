# Volatility Infrastructure Platform

A strategy-agnostic, production-grade options analytics platform implementing a complete volatility surface pipeline — from raw market data to risk-managed portfolios.

## Quick start for collaborators

```bash
pip install -r requirements.txt
npm install --prefix frontend
python start_dashboard.py
```

Open http://127.0.0.1:3000/ when the app is ready.

## Contributors

| Name | Role |
|------|------|
| **Enzo MERET** | Volatility infrastructure, pricing, risk & pipeline orchestration |
| **Nicolas GILHODES** | *Section à compléter* |
| **Pierre SALMON** | React/Vite frontend, FastAPI backend, UI-friendly pipeline outputs |
| **Yanis RAJHI** | *Section à compléter* |

---

# 📌 Contribution — Enzo MERET

## Architecture

```
Raw Events → Snapshot → Forward Curve → IV Solve → Surface Fit → Greeks → Risk
                 ↑            ↓              ↓           ↓          ↓
                QC         Carry          Diagnostics  Calendar    Scenarios
```

### Layer Responsibilities

| Layer | Module | Purpose |
|-------|--------|---------|
| **Connectivity** | `session.py` | IBKR state-machine session manager with reconnect |
| **Universe** | `contracts.py` | Instrument resolution, strike/expiry generation |
| **Collectors** | `raw_writer.py`, `synthetic_feed.py` | Event capture, synthetic data generation |
| **Snapshots** | `builder.py` | Deterministic market-state reconstruction |
| **Forwards** | `engine.py`, `carry.py` | Put-call parity forward curves, implied dividends |
| **QC** | `quote_quality.py`, `checks.py` | Named composable checks, system-level validation |
| **IV** | `solver.py` | Bracketed Brent + Newton IV inversion |
| **Surfaces** | `calibration.py`, `interpolation.py` | SVI calibration, cross-maturity interpolation |
| **Pricing** | `european.py`, `american.py` | Black-76, CRR binomial tree, Bjerksund-Stensland |
| **Risk** | `greeks.py`, `aggregation.py`, `scenarios.py` | Greeks, portfolio risk, stress testing |
| **Orchestration** | `jobs.py`, `replay.py` | Pipeline scheduling, historical replay |
| **Storage** | `schemas.py`, `writers.py`, `readers.py` | Frozen dataclasses, partitioned append-only I/O |
| **Utils** | `dates.py`, `math_helpers.py`, `logging_helpers.py` | Day-count, robust statistics, structured logging |

## Quick Start

```bash
# Run the full pipeline with synthetic data
python scripts/run_pipeline.py --date 2026-04-06 --underlying SPY --spot 450

# Dry run (no file outputs)
python scripts/run_pipeline.py --date 2026-04-06 --dry-run

# Stressed market regime
python scripts/run_pipeline.py --date 2026-04-06 --regime stressed --seed 99

# Export results to JSON
python scripts/run_pipeline.py --date 2026-04-06 --output-json results.json

# Historical replay
python scripts/run_pipeline.py --replay --start 2026-04-01 --end 2026-04-06
```

## Running Tests

```bash
# All test suites
python tests/test_pricing.py     # Put-call parity, Greeks, American pricing
python tests/test_solver.py      # IV solver convergence and diagnostics
python tests/test_forwards.py    # Forward curves and implied carry
python tests/test_surface.py     # SVI calibration and interpolation
python tests/test_qc.py          # Quote quality and validation checks
python tests/test_scenarios.py   # Scenario engine and risk attribution
```

## Key Design Decisions

1. **Frozen dataclasses everywhere** — All inter-module contracts use `@dataclass(frozen=True)`. No pandas in cross-module interfaces.

2. **Structured failures** — Every failed IV solve, QC rejection, or forward fallback returns a structured result with a reason code. No silent NaN.

3. **Same code path for live and replay** — The replay module calls `run_full_pipeline()` directly. No separate "historical" branch.

4. **Named, composable QC checks** — Each quality check is a separate function returning a structured `QcCheckResult`. No monolithic if-statements.

5. **Total variance space** — Surface calibration and interpolation operate in total variance (w = σ²T) for calendar-arbitrage awareness.

6. **Append-only storage** — Partitioned writes never overwrite. Malformed records are quarantined, never dropped.

## Configuration

All thresholds, scenario grids, and SLOs are defined in `configs/default.yaml` and versioned per the specification.

## Project Structure

```
project/
├── configs/
│   └── default.yaml           # Versioned configuration
├── scripts/
│   └── run_pipeline.py        # CLI entry point
├── src/
│   ├── collectors/            # Raw event capture
│   ├── connectivity/          # Broker session management
│   ├── forwards/              # Forward and carry engines
│   ├── iv/                    # Implied volatility solver
│   ├── orchestration/         # Pipeline scheduling and replay
│   ├── pricing/               # European and American pricing
│   ├── qc/                    # Quality control framework
│   ├── risk/                  # Greeks, aggregation, scenarios
│   ├── snapshots/             # Market-state reconstruction
│   ├── storage/               # Schemas, readers, writers
│   ├── surfaces/              # SVI calibration and interpolation
│   └── utils/                 # Dates, math, logging
└── tests/
    ├── test_pricing.py        # 13 tests
    ├── test_solver.py         # 8 tests
    ├── test_forwards.py       # 5 tests
    ├── test_surface.py        # 7 tests
    ├── test_qc.py             # 8 tests
    └── test_scenarios.py      # 7 tests
```

---

# 📌 Contribution — Nicolas GILHODES

*Section à compléter par Nicolas.*

---

# 📌 Contribution — Pierre SALMON

I added a simple React-based dashboard and a lightweight FastAPI backend so the volatility pipeline can be launched and explored from a browser.

## What was added

- A React/Vite frontend in the `frontend/` folder
- A FastAPI backend in `backend.py` that exposes the existing Python pipeline through HTTP
- Additional UI-friendly pipeline outputs for QC, risk, scenarios, replay, and configuration

## How to run the UI locally

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Install frontend dependencies

```bash
cd frontend
npm install
```

### 3) Start the backend

From the project root:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

### 4) Start the frontend

In a second terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Then open:

- http://localhost:3000/ for the UI
- http://localhost:8000/docs for the API documentation

## What the UI allows

- Run the volatility pipeline with custom parameters
- Review QC checks and risk metrics
- Explore scenario outcomes
- Replay the pipeline over a date range
- Inspect a configuration snapshot

This makes the project easier to demo and to discuss with collaborators who are not comfortable running the Python CLI directly.

---

# 📌 Contribution — Yanis RAJHI

*Section à compléter par Yanis.*
