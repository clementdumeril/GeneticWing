# GeneticWing

Evolutionary, flatness-based planning of energy-harvesting trajectories for
fixed-wing UAVs.

GeneticWing extends the ARPL flat planning pipeline for fixed-wing aircraft
(Morando et al., [ICRA 2025](https://arxiv.org/abs/2502.00581) and
[arXiv 2605.15619](https://arxiv.org/abs/2605.15619)) from efficient gliding
to soaring. An evolutionary search runs over Bernstein-polynomial position
trajectories. Differential flatness turns each candidate into airspeed, C_L,
bank and required thrust with no integration, so a whole population is scored
in one vectorized pass. The best candidate then seeds a local NLP and an
energy-aware NMPC.

## Status: phase 1 (simulation foundation)

| Module | Content |
| --- | --- |
| `geneticwing.bernstein` | Bernstein segments and chains, derivatives, batch evaluation |
| `geneticwing.aircraft` | Point-mass airframe, parabolic polar, exact steady glide |
| `geneticwing.flatness` | Flat map: trajectory + wind -> airspeed, C_L, bank, load factor, T_req |
| `geneticwing.energy` | Specific energy, netto variometer |
| `geneticwing.wind` | Level-N1 winds: uniform, thermal (Allen scaling), ridge lift, Dryden turbulence |
| `geneticwing.planning.glide` | Local NLP glide planner (baseline, SciPy SLSQP) |
| `geneticwing.simulation` | Closed-loop point-mass flight: path tracker + attitude lag |
| `scenarios/s0_glide.py` | Scenario S0: reproduction of the ARPL straight glide |

Unpowered, position alone is not a flat output: a glider has two inputs
(C_L, bank) for three outputs. The planner therefore treats `T_req = 0` as a
constraint, which generalizes the sink-rate constraint of arXiv 2605.15619 to
non-uniform wind.

## Scenario S0: reproduction of the ARPL glide

The setup is a 400 m leg from 120 m, with a 12.5 m/s airspeed reference and a
5.2 m/s tailwind. The closed-loop column is the mean of 50 Dryden realizations.

| Metric | ARPL flight | Plan, uniform wind | Closed loop, Dryden |
| --- | --- | --- | --- |
| Mean airspeed (m/s) | 12.50 | 12.50 | 12.61 |
| Mean sink rate (m/s) | 1.48 | 1.58 | 1.61 |
| Glide ratio | 9.83 | 11.15 | 11.27 |
| Airspeed RMSE (m/s) | 1.09 | 0.00 | 0.87 |
| Sink-rate RMSE (m/s) | 0.39 | 0.00 | 0.43 |
| Netto RMSE (m/s) | 0.69 | 0.00 | 0.72 |

The planner converges from 20 out of 20 perturbed initial guesses.

Known gaps, to close with the lab's data:

- Stratosurfer **mass (1.3 kg) and wing area (0.30 m²) are placeholders**. The
  polar is fitted to the best-glide point the paper reports (12 m/s, 1.51 m/s
  sink). As a result, the predicted sink at 12.5 m/s (1.58 m/s) sits above the
  measured 1.48 m/s.
- The paper gives the wind direction but not the leg direction. A pure
  tailwind is assumed.
- The netto filter window (1 s Savitzky-Golay) and Dryden W20 (= mean wind)
  are assumptions.
- The simulation is point-mass with a first-order attitude lag, not 6-DOF.
- The glide ratio is higher than measured: the model has no tracking losses
  beyond turbulence and no airframe imperfections.

## Usage

```bash
pip install -e ".[dev]"
python -m pytest
python scenarios/s0_glide.py --out results
```

Conventions: ENU frame (z up), SI units, wind is the air-mass velocity.
Arrays are `(..., N, 3)` so populations evaluate in one call.
