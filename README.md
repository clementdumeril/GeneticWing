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
| `geneticwing.seeding` | Fit a genome to any flight (the law's), to seed the planners |
| `scenarios/ga_tuning.py` | GA hyperparameter sweep on training instances (population 50 wins) |
| `scenarios/phase2_teacher.py` | The study: 3 scenarios x 3 test instances x 11 methods |

Cost is the net onboard energy over the horizon (motor energy minus the
specific energy stored), normalized by level cruise on the motor: `1` means
cruising on the motor, `< 0` means net energy gained from the air. Budget:
20k evaluations, 100k for the `@100k` rows. Mean over the 3 instances,
counting feasible solutions only.

| Method | S1 thermal | S2 field to goal | S3 ridge |
| --- | --- | --- | --- |
| NLP (straight start) | -2.77 | infeasible | -2.11 (1/3) |
| NLP multi-start | 0.96 (2/3) | infeasible | -2.35 (1/3) |
| GA (tuned, population 50) | 0.15 | infeasible | -0.73 |
| Zohdi GA | infeasible | infeasible | infeasible |
| CMA-ES | -0.32 | infeasible | -1.92 |
| CMA-ES + NLP | -3.80 | infeasible | **-2.16** |
| GA @100k | -0.99 | 0.47 | -1.83 |
| CMA-ES @100k | -2.14 | 0.78 | -2.10 |
| Reactive law (Zohdi-tuned) | -4.67 | **-0.15** | -1.98 |
| **Law + NLP** (seeded) | **-6.08** | infeasible | 0.89 |
| Law + CMA-ES + NLP (seeded) | -4.44 | infeasible | -2.05 |

![Trajectories](docs/figures/phase2_trajectories.png)
![Convergence](docs/figures/phase2_convergence.png)

What this says about H1:

- **H1 is not supported as stated.** Even tuned, the real-coded GA is the
  weakest of the working planners at equal budget, and the course's GA (80 %
  random immigrants each generation) finds nothing feasible in 87 to 159
  dimensions.
- **Structure beats search.** The reactive law, tuned offline Zohdi-style,
  encodes the orbit the planners must otherwise discover. Fitting a genome to
  its flight and refining it with the NLP gives the best results on S1
  (-6.08, every instance), 30 % more energy than the law alone.
- **S2 stays out of reach of the planners at 20k**: the law is the only method
  that soars there without the motor. The fixed 150 s horizon with a terminal
  goal over 159 genes is the likely culprit; receding horizon is the fix.
- On S3 the law's flight is fitted less accurately (20 m), and the NLP from it
  ends in a poor optimum; CMA-ES + NLP remains the best there.
- **The teacher is a portfolio** (law + NLP, the law, CMA-ES + NLP), as planned
  in the risk table of the design document. The student does not depend on
  which optimizer produced the data.

Caveats: 3 instances per scenario; point-mass model; the law knows where the
lift is, as the planners do.

## Phase 3: generative student (hypotheses H2, H3, H4)

| Module | Content |
| --- | --- |
| `geneticwing.benchmarks.s1_family` | Randomized single-thermal family (thermal strength, wind, start); heading-frame context and exact left/right mirror |
| `geneticwing.benchmarks.with_margins` | Tightened constraints for the teacher (constraint tightening) |
| `geneticwing.student` | Conditional flow matching (rectified flow), deterministic regression baseline, learned mutation, smoothing |
| `scenarios/phase3_dataset.py` | Teacher dataset: law flight -> fitted genome -> NLP, both orbit modes, 600 train + 80 test instances |
| `scenarios/phase3_student.py` | Training and evaluation against the teacher on held-out instances |

Teacher: 1263 solutions; 593 instances with both orbit modes, 77 with one.
Mean cost -5.43 under the nominal constraints, against -3.97 for the law alone.
Students: MLP 4 x 512, 20k steps on CPU, trained on the data and its mirror.
Evaluation on 78 held-out instances (teacher mean cost -5.19):

| Method | Instances solved | Samples feasible | Mean cost | Gap to teacher | Median time |
| --- | --- | --- | --- | --- | --- |
| Regression (deterministic) | 0 % | 0 % | - | - | 1 ms |
| Flow matching, 64 samples, 1 step | 90 % | 17 % | 0.87 | 6.11 | 28 ms |
| Flow matching, 64 samples, 8 steps | 67 % | 15 % | -2.06 | 3.11 | 49 ms |
| Flow matching + learned mutations | 74 % | 16 % | -1.95 | 3.31 | 137 ms |
| Flow matching + mutations + NLP (2000 evals) | 76 % | 75 % | -2.51 | 2.72 | 1.0 s |
| Reactive law | 100 % | 100 % | -3.69 | 1.50 | feedback law |
| NLP from straight flight (2000 evals) | 8 % | 8 % | 1.49 | 6.95 | 0.9 s |

![Student](docs/figures/phase3_student.png)

What this says:

- **H3 holds.** The regression network averages the clockwise and
  counter-clockwise orbits into a straight line out of the thermal (red dashes)
  and never produces a feasible trajectory. Flow matching samples circle in
  the thermal, in both directions.
- **H2 holds only in part.** The student plans in 30 to 50 ms on a laptop CPU
  and finds a feasible soaring trajectory for 67 to 90 % of new instances, but
  keeps only about 40 % of the teacher's energy gain, and does not yet beat the
  reactive law it was partly distilled from.
- **H4 holds in part.** The NLP warm-started by the student solves 76 % of the
  instances, against 8 % for the same NLP from straight flight, and narrows
  the gap; learned mutations bring nothing measurable here.
- **Feasibility is the bottleneck, as anticipated.** A first teacher solved to
  the nominal constraints sat exactly on them: keeping 99.4 % of the genome's
  variance (48 principal components) already broke 80 % of its solutions, and
  that student solved 25 to 40 % of instances with 3 % feasible samples.
  Solving the teacher with margins (cost: 0.24 of the 5.4 energy units) made
  its solutions 8 to 16 times more tolerant to gene noise and gave the numbers
  above; smoothing the samples took feasibility from 2 to 7 % in a diagnostic.

Caveats: about 1.3k teacher solutions for 87 genes is little data; CPU
training; point-mass model; the law's time is a simulation cost, as a feedback
law it runs in microseconds.

Next steps: 10x more teacher data on the cluster, a transformer over control
points (as in DynFly) or latent flow matching, constraint-aware training,
DAgger and the self-improvement loop (H6), and the closed-loop evaluation of
phase 4.

## Usage

```bash
pip install -e ".[dev]"
python -m pytest
python scenarios/s0_glide.py --out results
python scenarios/phase2_teacher.py --out results/phase2 --workers 8
python scenarios/phase3_dataset.py --train 600 --test 80
python scenarios/phase3_student.py --steps 20000
```

## Running phase 3 at scale on Savio (Berkeley)

The teacher's dataset is the expensive part (about 20 core-seconds per
instance) and splits into independent shards; the students train on GPU.
`slurm/submit_all.sh` chains four jobs: tune the law once, solve the shards
(one node each, plus one for the test set), merge, then train and evaluate one
student per training-set size (the scaling study). With the defaults, 15k
training instances cost about 80 core-hours of CPU and five short GPU jobs.

On a login node, once:

```bash
git clone https://github.com/clementdumeril/GeneticWing.git
cd GeneticWing
bash slurm/check_savio.sh      # prints your accounts, partitions, QoS and GPU types
nano slurm/config.sh           # set ACCOUNT, the CPU/GPU partitions and QoS, the Python module
bash slurm/setup_env.sh        # venv in /global/scratch/users/$USER/geneticwing
```

Then:

```bash
bash slurm/submit_all.sh       # law -> shards -> merge -> students
squeue -u $USER
bash slurm/submit_all.sh train # later: retrain only, on the merged dataset
```

Results land in `/global/scratch/users/$USER/geneticwing/phase3/students/n_<size>/`
and logs in `.../phase3/logs/`. Sizes, shard counts, budgets and walltimes are
all in `slurm/config.sh`.

Conventions: ENU frame (z up), SI units, wind is the air-mass velocity.
Arrays are `(..., N, 3)` so populations evaluate in one call.
