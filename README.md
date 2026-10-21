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

## Phase 1: simulation foundation

| Module | Content |
| --- | --- |
| `geneticwing.bernstein` | Bernstein segments and chains, derivatives, batch evaluation |
| `geneticwing.aircraft` | Point-mass airframe, parabolic polar, exact steady glide; Stratosurfer (ARPL platform, placeholder mass and area) and SB-XC sailplane (estimated polar) |
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

## Phase 2: offline teacher (hypothesis H1)

| Module | Content |
| --- | --- |
| `geneticwing.spline` | Clamped B-splines (chains of Bernstein pieces) with initial-state control points |
| `geneticwing.soaring` | Soaring problem: turtle-style genome (turn rate, ground speed, path angle), energy fitness, hinge constraints, Deb ranking |
| `geneticwing.benchmarks` | S1 single thermal, S2 thermal field to a goal, S3 ridge lift, on the SB-XC with a 15 N motor |
| `geneticwing.evolution` | Real-coded GA, the GA of T. Zohdi's course, CMA-ES; equal-budget interface |
| `geneticwing.planning.local` | SLSQP on the same genome, single and multi-start, batched finite differences |
| `geneticwing.policies` | Reactive orbit law (10 parameters) tuned offline by the Zohdi GA |
| `scenarios/phase2_teacher.py` | The study: 3 scenarios x 3 test instances x 9 methods |

Cost is the net onboard energy over the horizon (motor energy minus the
specific energy stored), normalized by level cruise on the motor: `1` means
cruising on the motor, `< 0` means net energy gained from the air. Budget:
20k evaluations, 100k for the `@100k` rows. Mean over the 3 instances,
counting feasible solutions only.

| Method | S1 thermal | S2 field to goal | S3 ridge |
| --- | --- | --- | --- |
| NLP (straight start) | -2.77 (3/3) | infeasible | -2.11 (1/3) |
| NLP multi-start | 0.96 (2/3) | infeasible | -2.35 (1/3) |
| GA | 0.48 | infeasible | -0.15 |
| Zohdi GA | infeasible | infeasible | infeasible |
| CMA-ES | -0.32 | infeasible | -1.92 |
| **CMA-ES + NLP** | **-3.80** | infeasible | **-2.16** |
| GA @100k | -1.44 | 0.96 | -1.72 |
| CMA-ES @100k | -2.14 | 0.78 | -2.10 |
| **Reactive law (Zohdi-tuned)** | **-4.67** | **-0.15** | -1.98 |

![Trajectories](docs/figures/phase2_trajectories.png)
![Convergence](docs/figures/phase2_convergence.png)

What this says about H1:

- **H1 is not supported as stated.** The real-coded GA is the weakest of the
  working planners at equal budget, and the course's GA (80 % random
  immigrants each generation) finds nothing feasible in 87 to 159 dimensions.
- **CMA-ES followed by the local NLP is the best planner**, and the reactive law
  tuned offline is the best method overall on S1 and S2. The law encodes the
  orbit structure the planners have to discover.
- **S2 is hard for every planner**: only the 100k-budget runs reach the goal,
  and they still need the motor. The fixed horizon with a terminal goal over
  159 genes is the likely culprit (see next steps).
- For the teacher of phase 3 this means: **the teacher is a portfolio**
  (CMA-ES + NLP, the tuned law), as planned in the risk table of the design
  document. The student does not depend on which optimizer produced the data.

Caveats: GA hyperparameters are defaults, not tuned; 3 instances per scenario;
point-mass model; the law knows where the lift is, as the planners do.

Next steps for phase 2: seed the planners with the law's flights (fit a genome
to them, then CMA-ES + NLP), receding-horizon S2 instead of one 150 s plan,
tuned GA settings, and more instances.

## Usage

```bash
pip install -e ".[dev]"
python -m pytest
python scenarios/s0_glide.py --out results
python scenarios/phase2_teacher.py --out results/phase2 --workers 8
```

Conventions: ENU frame (z up), SI units, wind is the air-mass velocity.
Arrays are `(..., N, 3)` so populations evaluate in one call.
