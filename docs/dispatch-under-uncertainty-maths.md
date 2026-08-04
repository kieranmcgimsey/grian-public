# Dispatch under uncertainty: from point forecasts to CVaR — with the maths

A teaching note on *how* a battery should turn an uncertain price forecast into a
trade. It builds a ladder of objectives — point → worst-case → expected-value →
mean-CVaR → two-stage recourse — with the maths, a worked example for each, and an
honest note on which are actually implemented in grian. Pairs with
`probabilistic-dispatch-explained.md` and `quantile-mpc-postmortem.md`.

---

## 1. The decision problem

A battery over a horizon of $T$ intervals of length $\Delta t$. At each step we
choose charge $c_t\ge 0$ and discharge $d_t\ge 0$ (MW). The state of charge evolves
as $\mathrm{SOC}_t = \mathrm{SOC}_{t-1} + \eta\,\Delta t\, c_t - \tfrac{\Delta t}{\eta} d_t$,
with $\eta=\sqrt{0.85}$, bounded $0\le \mathrm{SOC}_t\le E$, power $0\le c_t,d_t\le P$,
and a daily throughput (cycle) cap. Call this feasible set $\mathcal{A}$.

The future price is **uncertain**, described by a forecast *fan* — a set of
scenarios $s=1,\dots,S$ with price paths $p^s = (p^s_1,\dots,p^s_T)$ and
probabilities $\pi_s$ (from the quantile levels; e.g. $q_{05},q_{50},q_{90},q_{98}$).

Revenue of an action plan $a=(c,d)$ under scenario $s$:

$$
R_s(a) \;=\; \sum_{t=1}^{T} p^s_t\,(d_t - c_t)\,\Delta t .
$$

The whole question is: **which functional of $\{R_s(a)\}_s$ do we maximise?** That
single choice is the difference between whipsawing, freezing, and trading well.

---

## 2. The ladder of objectives

### (0) Point forecast — what plain MPC does

Collapse the fan to one path $\hat p$ (mean or median) and solve a deterministic LP:

$$
\max_{a\in\mathcal{A}} \; \sum_t \hat p_t (d_t-c_t)\,\Delta t .
$$

**Uncertainty is thrown away.** A confident-but-wrong forecast is acted on at full
size. *Example:* $\hat p$ says a \$200 evening peak → charge now at \$100. The peak
fizzles to \$60 → you bought high. This is grian's failing baseline (`mpc30`).
**Status: built.**

### (1) Worst-case / robust — "max–min"

Pick the plan whose *worst* scenario is best:

$$
\max_{a\in\mathcal{A}} \; \min_{s} \; R_s(a) .
$$

Infinitely risk-averse. *Example:* only charge if **even the pessimistic $q_{05}$
path** shows a peak big enough to clear the ~15% efficiency tax. Result: it almost
never trades, but it never whipsaws.

grian's `mpc_scenario` is a **tractable heuristic** for this: solve the LP on each
scenario separately (each path is internally coherent, so rankings are preserved),
then take $c_t=\min_s c^s_t,\; d_t=\min_s d^s_t$ — "act only where all scenarios
agree." That's why it under-trades (huge spread, low volume). **Status: built,
validating.**

### (2) Expected value — risk-neutral stochastic

Weight scenarios by probability:

$$
\max_{a\in\mathcal{A}} \; \mathbb{E}_s\!\left[R_s(a)\right] \;=\; \max_{a}\;\sum_s \pi_s R_s(a).
$$

A trade fires if profitable **on average** across the fan. Less timid than
worst-case. *But* it's blind to the tail: a plan that's great on average yet
catastrophic in 10% of scenarios is allowed. *Example:* charge if the
probability-weighted peak beats $ \hat p_{\text{now}}/\eta^2 $. **Status: not built
(it's the natural "soften the min" first step — see §4).**

### (3) Mean–CVaR — the tunable middle (the target)

Trade off average reward against **tail risk**. With loss $\ell_s = -R_s(a)$:

$$
\max_{a\in\mathcal{A}} \; \mathbb{E}[R_s(a)] \;-\; \lambda\,\mathrm{CVaR}_\alpha(\ell).
$$

$\mathrm{CVaR}_\alpha$ ("conditional value-at-risk") is the **mean loss in the worst
$\alpha$ fraction** of scenarios (e.g. worst 20%). $\lambda$ dials risk appetite:
$\lambda=0$ recovers expected-value (2); $\lambda\to\infty$ recovers worst-case (1).

The magic is that CVaR is an LP via the **Rockafellar–Uryasev** identity — introduce
a scalar $\eta$ (the value-at-risk level) and slacks $z_s\ge 0$:

$$
\mathrm{CVaR}_\alpha(\ell) \;=\; \min_{\eta}\;\Big\{\, \eta + \tfrac{1}{\alpha}\textstyle\sum_s \pi_s z_s \;:\; z_s \ge \ell_s - \eta,\; z_s\ge 0 \,\Big\}.
$$

So the whole dispatch becomes **one linear program**:

$$
\max_{a\in\mathcal{A},\,\eta,\,z\ge 0}\;\; \sum_s \pi_s R_s(a) \;-\; \lambda\Big(\eta + \tfrac{1}{\alpha}\sum_s \pi_s z_s\Big)
\quad\text{s.t.}\quad z_s \ge -R_s(a) - \eta .
$$

*Example ($\alpha=0.2,\ \lambda$ moderate):* "Charge if it's good on average **and**
not disastrous in the worst 20% of scenarios." It takes the solid opportunities and
declines the ones with a nasty downside — the Goldilocks between freezing and
whipsawing. **Status: not built (this is the principled `v2-prob`).**

### (4) Two-stage with recourse — the honest MPC form

MPC only *executes the current block* before re-solving. Reality: you must commit
**one** action *now* (the present is shared across all futures — "non-anticipativity"),
but you may **adapt** later once you learn which scenario you're in. Let $a_0$ be the
here-and-now action and $a^s_{>0}$ the scenario-specific recourse:

$$
\max_{a_0,\,\{a^s_{>0}\},\,\eta,\,z}\;\; \sum_s \pi_s R_s(a_0, a^s_{>0}) \;-\; \lambda\Big(\eta + \tfrac1\alpha\sum_s \pi_s z_s\Big)
$$

subject to per-scenario SOC dynamics, the shared-$a_0$ constraint, and the CVaR
rows. This is the *correct* stochastic MPC: hedge the irreversible present decision,
plan to react in the future. It's a bigger LP ($\approx S\times$ variables) but still
linear. **Status: not built.**

---

## 3. The fan must be calibrated (or all of this is garbage)

Every objective above trusts the quantiles. If the model says "$q_{90}=\$400$" but
actual prices exceed \$400 **30%** of the time, the fan lies and every hedge
misfires. Two checks:

- **Coverage.** For a calibrated fan, $\Pr[\,y \le q_\tau\,]=\tau$. Plot empirical
  coverage vs nominal $\tau$; it should sit on the diagonal.
- **CRPS** (continuous ranked probability score) — a *proper* score for the whole
  distribution, not just the mean:
  $$
  \mathrm{CRPS}(F,y) \;=\; \int_{-\infty}^{\infty}\big(F(x) - \mathbb{1}[y\le x]\big)^2\,dx .
  $$
- **Conformal prediction** — a post-hoc widening/narrowing of the fan using recent
  residual quantiles that gives a **finite-sample coverage guarantee**. Cheap, and
  the right thing to bolt on before trusting the CVaR hedge.

*Example:* if the fan is systematically too narrow, the CVaR term under-estimates
the tail → it hedges too little → you're back to whipsawing. Calibrate first.

---

## 4. Comparison

| Approach | Objective | Risk stance | Behaviour | Status |
|---|---|---|---|---|
| Point (`mpc30`) | $\max R(\hat p)$ | ignores uncertainty | over-trades / whipsaws | built |
| Worst-case (`mpc_scenario` ≈) | $\max_a\min_s R_s$ | maximally averse | under-trades, never whipsaws | built |
| Expected value | $\max \mathbb{E}[R_s]$ | risk-neutral | balanced, tail-blind | **next** |
| **Mean–CVaR** | $\max \mathbb{E}[R]-\lambda\,\mathrm{CVaR}$ | **tunable via $\lambda$** | **Goldilocks** | roadmap |
| Two-stage recourse | + non-anticipativity | tunable | correct stochastic MPC | roadmap |

**The planned path:** (a) **soften the `min`** in `mpc_scenario` to the
probability-weighted expected value (2) — smallest change, likely fixes the
under-trading; (b) add the **CVaR term** with a $\lambda$ knob (3) — one LP via
Rockafellar–Uryasev; (c) if worthwhile, the **two-stage recourse** LP (4); and
(0.5) **calibrate the fan** (conformal + CRPS) *before* trusting any hedge. Each is
a clean, self-contained addition to `dispatch_prob.py`.
