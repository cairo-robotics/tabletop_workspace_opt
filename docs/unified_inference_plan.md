# Unified Intent Inference: Trajectory-Level Gaussian-Boltzmann Model

A plan for combining Gaussian joystick noise modeling with Boltzmann
path efficiency into a single intent inference framework.

---

## 1. Background: Two Existing Approaches

### 1.1 Approach A: Gaussian Direction Model (Current Intent Separability)

**What it does.** At each timestep, the robot observes a noisy
joystick command and asks: "which goal is this direction consistent
with?"

**User model.** The user points the joystick toward their intended
goal. The observed command is the unit direction plus Gaussian noise:

$$
u_t = \frac{g^* - x_t}{\|g^* - x_t\|} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

**Inference.** Each candidate goal $g$ has an expected direction
$\mu_t(g) = (g - x_t) / \|g - x_t\|$. The robot accumulates
Gaussian log-likelihoods:

$$
\log S_t(g) = \log S_{t-1}(g) - \frac{1}{2}(u_t - \mu_t(g))^\top \Sigma^{-1} (u_t - \mu_t(g))
$$

**Limitation.** Each observation $u_t$ is treated independently.
The model doesn't consider the trajectory shape — it only asks
"what direction is the joystick pointing right now?" A user who
takes a detour but is clearly heading toward a goal won't be
recognized until the joystick points directly at it.

### 1.2 Approach B: Path Efficiency Model (Dragan Legibility)

**What it does.** The robot looks at the entire trajectory so far
and asks: "which goal would this trajectory be an efficient path
toward?"

**Observer model.** Given trajectory prefix $\xi_{0:t}$, the
observer computes a Boltzmann-rational posterior:

$$
P(g \mid \xi_{0:t}) = \frac{\exp(-\beta \cdot C(\xi_{0:t}, g))}{\sum_{g'} \exp(-\beta \cdot C(\xi_{0:t}, g'))}
$$

where the cost is the path efficiency ratio:

$$
C(\xi_{0:t}, g) = \frac{L(\xi_{0:t}) + d(x_t, g)}{d(x_0, g)}
$$

- $L(\xi_{0:t})$ = total path length traveled (sum of step distances)
- $d(x_t, g)$ = remaining straight-line distance from current EE to goal $g$
- $d(x_0, g)$ = initial straight-line distance from start to goal $g$

A perfectly efficient trajectory toward $g$ has cost = 1.0 at all
times (the numerator equals the denominator). Detours increase cost.

**Limitation.** No explicit noise model. The cost ratio is
deterministic given the trajectory. When $d(x_0, g) \approx 0$
(goal is near the start position), the cost blows up regardless
of trajectory quality (the $d_{start} \approx 0$ singularity).

### 1.3 Why Neither Is Complete

- **Gaussian direction** ignores trajectory shape. It can't reason
  about "this path has been heading northeast for 20 steps, so the
  goal is probably in the northeast." It only sees the current
  direction.

- **Path efficiency** ignores noise structure. It doesn't model
  *why* the trajectory deviates from the straight line. A noisy
  but goal-directed trajectory looks the same as a genuinely
  inefficient one.

---

## 2. The Unified Approach: Gaussian Noise on Trajectories + Boltzmann Observer

### 2.1 Core Idea

Model the user's joystick commands as Gaussian-noisy, propagate
that noise through the trajectory dynamics, and then evaluate the
resulting noisy trajectory using the Boltzmann path efficiency
observer. This gives us:

- A **generative model** of how trajectories are produced (Gaussian
  joystick noise integrated over time)
- An **observer model** of how trajectories are interpreted
  (Boltzmann cost efficiency)
- A **certificate** for correct goal inference (linearized margin
  under Gaussian trajectory noise)

### 2.2 Step 1: Noisy Trajectory Generation

The user sends joystick commands with Gaussian noise:

$$
u_t = \bar{u}_t(g^*) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma_u)
$$

where $\bar{u}_t(g^*)$ is the ideal command (e.g., unit direction
toward $g^*$ scaled by speed). The EE integrates these commands:

$$
x_{t+1} = x_t + u_t \cdot \Delta t
$$

After $T$ steps, the realized trajectory is:

$$
\xi = (x_0, x_1, \dots, x_T)
$$

Because each $u_t$ has independent Gaussian noise, the trajectory
$\xi$ is a random variable. Around the nominal (noise-free)
trajectory $\bar{\xi}$:

$$
\xi \approx \bar{\xi}(g^*) + \eta, \qquad \eta \sim \mathcal{N}(0, \Sigma_\xi)
$$

where $\Sigma_\xi$ is the covariance of the trajectory deviation,
determined by the noise covariance $\Sigma_u$ propagated through the
dynamics. For simple integrator dynamics ($x_{t+1} = x_t + u_t \Delta t$),
the position at time $t$ accumulates noise from all previous steps:

$$
x_t - \bar{x}_t = \Delta t \sum_{s=0}^{t-1} \varepsilon_s
$$

So later positions have more accumulated noise (random walk).

### 2.3 Step 2: Boltzmann Observer on the Noisy Trajectory

The observer sees the realized (noisy) trajectory $\xi$ and computes
the posterior using the path efficiency cost:

$$
P(g \mid \xi) \propto \exp\!\left(-\beta \cdot \frac{L(\xi) + d(x_t, g)}{d(x_0, g)}\right)
$$

The observer doesn't know the noise model — it just evaluates how
efficient the trajectory looks for each goal. But because the
trajectory is random, the posterior is also random.

### 2.4 Step 3: Pairwise Margin

For correct inference, we need the true goal $g^*$ to have higher
posterior than all alternatives. Define the pairwise log-posterior
margin:

$$
M_g(\xi) = \log \frac{P(g^* \mid \xi)}{P(g \mid \xi)} = \beta \left(C(\xi, g) - C(\xi, g^*)\right) + \log \frac{P(g^*)}{P(g)}
$$

Correct inference requires $M_g(\xi) \geq 0$ for all $g \neq g^*$.

---

## 3. The Linearized Margin

### 3.1 Why Linearize?

The cost $C(\xi, g)$ is a nonlinear function of the trajectory $\xi$
(it involves path length $L(\xi)$, which is a sum of norms, and
distance $d(x_t, g)$, which is also a norm). Even though $\xi$ is
approximately Gaussian, $M_g(\xi)$ is not Gaussian because of these
nonlinearities.

To get an analyzable (Gaussian) margin, we **linearize** the cost
difference around the nominal trajectory $\bar{\xi}$.

### 3.2 The Linearization

Define the cost difference:

$$
\Delta C_g(\xi) = C(\xi, g) - C(\xi, g^*)
$$

First-order Taylor expansion around $\bar{\xi}$:

$$
\Delta C_g(\xi) \approx \Delta C_g(\bar{\xi}) + a_g^\top (\xi - \bar{\xi})
$$

where $a_g = \nabla_\xi \Delta C_g(\xi) \big|_{\xi = \bar{\xi}}$ is
the gradient of the cost difference with respect to the trajectory,
evaluated at the nominal trajectory.

### 3.3 What $a_g$ Captures

The gradient $a_g$ tells us: "if the trajectory deviates from the
nominal by a small amount $\delta\xi$, how much does the cost
difference between goals $g$ and $g^*$ change?"

For the path efficiency cost $C(\xi, g) = (L(\xi) + d(x_t, g)) / d(x_0, g)$:

- Deviations that increase path length $L(\xi)$ increase cost for
  all goals equally (path length doesn't depend on $g$)
- Deviations that move $x_t$ closer to $g$ decrease $d(x_t, g)$,
  reducing cost for $g$
- So $a_g$ primarily reflects how trajectory deviations affect the
  *remaining distance* to each goal differently

### 3.4 The Linearized Margin is Gaussian

Substituting the linearized cost difference into the margin:

$$
M_g(\xi) \approx \underbrace{\beta \cdot \Delta C_g(\bar{\xi}) + \log \frac{P(g^*)}{P(g)}}_{m_g \text{ (mean margin)}} + \beta \cdot a_g^\top \underbrace{(\xi - \bar{\xi})}_{\eta \sim \mathcal{N}(0, \Sigma_\xi)}
$$

Since $\eta$ is Gaussian and the margin is an affine function of
$\eta$, the margin is also Gaussian:

$$
M_g(\xi) \sim \mathcal{N}(m_g, \; v_g)
$$

with:

$$
m_g = \beta \cdot \Delta C_g(\bar{\xi}) + \log \frac{P(g^*)}{P(g)}
$$

$$
v_g = \beta^2 \cdot a_g^\top \Sigma_\xi \, a_g
$$

### 3.5 Interpretation

- **$m_g$ (mean margin):** How much better the nominal (noise-free)
  trajectory looks for $g^*$ compared to $g$, scaled by $\beta$.
  Larger $m_g$ = the trajectory is clearly more efficient for the
  true goal. This depends on the **layout** — well-separated objects
  produce larger cost differences.

- **$v_g$ (margin variance):** How much the margin fluctuates due to
  joystick noise. Larger $v_g$ = the noise could plausibly make the
  trajectory look efficient for the wrong goal. This depends on the
  **noise level** and the **sensitivity** of the cost difference to
  trajectory deviations.

---

## 4. The Sufficient Condition (Certificate)

### 4.1 From Gaussian Margin to Probability Bound

Since $M_g \sim \mathcal{N}(m_g, v_g)$, we can compute the
probability of correct inference:

$$
P[M_g \geq 0] = \Phi\!\left(\frac{m_g}{\sqrt{v_g}}\right)
$$

where $\Phi$ is the standard normal CDF.

### 4.2 The Slack Condition

To guarantee $P[M_g \geq 0] \geq 1 - \alpha_g$, we need:

$$
\frac{m_g}{\sqrt{v_g}} \geq \Phi^{-1}(1 - \alpha_g)
$$

Rearranging:

$$
m_g \geq \Phi^{-1}(1 - \alpha_g) \cdot \sqrt{v_g}
$$

This is the **slack condition** — the mean margin must exceed a
noise-adjusted threshold. The "slack" is:

$$
\text{slack}_g = m_g - \Phi^{-1}(1 - \alpha_g) \cdot \sqrt{v_g}
$$

Positive slack guarantees correct pairwise inference with
probability $\geq 1 - \alpha_g$.

### 4.3 Global Guarantee

By union bound over all $g \neq g^*$, with $\sum \alpha_g \leq \alpha$:

$$
P[\text{correct inference}] = P\!\left[\forall g \neq g^*: M_g \geq 0\right] \geq 1 - \alpha
$$

### 4.4 Workspace Optimization Objective

Maximize the worst-case slack:

$$
\max_\theta \; \min_{g \neq g^*} \left(m_g(\theta) - \Phi^{-1}(1 - \alpha_g) \cdot \sqrt{v_g(\theta)}\right)
$$

This has **exactly the same form** as the intent separability
objective, but with $m_g$ and $v_g$ derived from the Boltzmann
cost difference instead of the Gaussian direction difference.

---

## 5. Comparison to Intent Separability

The original intent separability framework does essentially the
same thing, but at the **command level** rather than the
**trajectory level**:

| Component | Intent Separability | Trajectory-Level (This) |
|-----------|-------------------|------------------------|
| Random variable | $u_t$ (single command) | $\xi$ (full trajectory) |
| Noise | $\varepsilon_t \sim \mathcal{N}(0, \sigma^2 I)$ per step | $\eta \sim \mathcal{N}(0, \Sigma_\xi)$ on trajectory |
| Likelihood model | Gaussian on direction | Boltzmann on path efficiency |
| Margin | $\sum_t \Delta_t^\top \Sigma^{-1} \Delta_t$ (Mahalanobis) | $\beta \cdot \Delta C_g(\bar{\xi})$ (cost difference) |
| Variance | $\tilde{V}(g)$ (cumulative direction separation) | $\beta^2 a_g^\top \Sigma_\xi a_g$ (cost sensitivity to noise) |
| Slack form | $m_g - \Phi^{-1}(1-\alpha_g)\sqrt{v_g}$ | Same form, different $m_g$ and $v_g$ |

The key insight: **both produce the same type of slack condition**,
just derived from different observation models. The trajectory-level
version is richer because it considers the cumulative path shape
rather than individual commands.

---

## 6. Online Predictor

For use in the shared autonomy runner, the online predictor would:

1. At each timestep, observe the actual EE position $x_t$
2. Compute the full trajectory prefix $\xi_{0:t} = (x_0, \dots, x_t)$
3. For each candidate goal $g$, compute the path efficiency cost:

$$
C(\xi_{0:t}, g) = \frac{L(\xi_{0:t}) + d(x_t, g)}{d(x_0, g)}
$$

4. Compute the Boltzmann posterior:

$$
P(g \mid \xi_{0:t}) = \frac{\exp(-\beta \cdot C(\xi_{0:t}, g))}{\sum_{g'} \exp(-\beta \cdot C(\xi_{0:t}, g'))}
$$

5. When $\max_g P(g) \geq \text{threshold}$, declare that goal
   as identified

This is the same as the existing `PathEfficiencyInference`, but
now we understand it as the **Boltzmann observer applied to a
Gaussian-noisy trajectory**, and the optimization objective
(linearized margin slack) provides a theoretical guarantee that
the observer will converge correctly.

### 6.1 Handling the $d_{start} \approx 0$ Singularity

When $d(x_0, g)$ is very small, the cost $C$ blows up. In the
linearized framework, this manifests as a very large $a_g$
(high sensitivity to noise), which makes $v_g$ large and the
slack negative — correctly indicating that inference is unreliable
for nearby objects.

The optimization objective naturally avoids placing objects near
the EE because the slack would be poor. This is equivalent to
the robot exclusion constraint we added manually, but now it
falls out of the theory.

---

## 7. Implementation Plan

### 7.1 Compute $a_g$ and $\Sigma_\xi$ for the Linearized Margin

- $\Sigma_\xi$: for integrator dynamics with noise $\sigma_u$ per
  step, position $x_t$ has variance $t \cdot \sigma_u^2 \cdot \Delta t^2$.
  The full trajectory covariance is block-diagonal with increasing
  variance.

- $a_g = \nabla_\xi \Delta C_g$: can be computed analytically or
  via finite differences. For path efficiency cost, the gradient
  involves derivatives of path length and remaining distance with
  respect to each trajectory point.

### 7.2 New Optimization Objective

Implement `trajectory_margin_objective()` that:
1. Simulates the nominal trajectory toward each goal
2. Computes $m_g$ (mean cost difference × beta + log prior ratio)
3. Computes $v_g$ (gradient × trajectory covariance × gradient)
4. Returns worst-case slack: $\min_g (m_g - \Phi^{-1}(1-\alpha_g)\sqrt{v_g})$

### 7.3 Evaluation

- Run MAP-Elites with the trajectory margin objective
- Evaluate with the path efficiency predictor in the headless SA
- Compare against separability and legibility objectives

---

## 8. Open Questions

1. **Is the linearization accurate enough?** The path efficiency
   cost has norm operations (path length, distances) that are not
   smooth at zero. How well does the first-order approximation hold
   for typical noise levels?

2. **Should the online predictor use the linearized margin or the
   raw Boltzmann cost?** The linearized margin is for the
   *optimization certificate* — the online predictor could still
   use the exact (nonlinear) Boltzmann cost.

3. **How does $\beta$ interact with $\sigma_u$?** Higher $\beta$
   amplifies both the mean margin and the variance. Is there an
   optimal $\beta$ for a given noise level?
