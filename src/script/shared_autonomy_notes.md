\documentclass{article}
\usepackage{graphicx} % Required for inserting images
\usepackage{amsmath}
\usepackage{amssymb}

\title{Shared Autonomy Notes}
\author{yitu9550 }
\date{February 2026}

\begin{document}

\maketitle

\section{Shared autonomy with noisy joystick inputs and intent-identification guarantees}

\subsection{State, goals, and joystick measurements}
Let $x_t \in \mathcal{X}$ denote the system state at time $t$ (e.g., end-effector pose), and let the user have a discrete set of candidate intentions (goals)
\[
\mathcal{G} = \{g_1,\dots,g_M\}.
\]
Let the human provide a joystick command $u_t \in \mathbb{R}^d$ (e.g., desired Cartesian velocity).
We model the observation as a noisy measurement of an underlying mean command $\mu_t$:
\[
u_t = \mu_t + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\Sigma),
\]
where $\Sigma \succeq 0$ is the (unknown) command noise covariance. We assume a bounded-noise regime:
\[
\Sigma \preceq \bar{\Sigma}
\]
for some design threshold $\bar{\Sigma}\succ 0$ (Loewner order).

\subsection{Goal-conditioned joystick model}
Assume that under goal $g \in \mathcal{G}$, the human's mean command is predicted by a goal-conditioned policy
\[
\mu_t(g) = \pi_h(x_t,g)\in\mathbb{R}^d.
\]
Then the likelihood of observing $u_t$ under goal $g$ is
\[
p(u_t \mid x_t,g,\Sigma) = \mathcal{N}\!\bigl(u_t;\ \mu_t(g),\ \Sigma\bigr).
\]
For a horizon of $T$ steps, define $U_{0:T-1} := (u_0,\dots,u_{T-1})$. With conditional independence given $(x_t,g)$,
\[
p(U_{0:T-1} \mid x_{0:T-1}, g, \Sigma)
= \prod_{t=0}^{T-1} \mathcal{N}\!\bigl(u_t;\ \mu_t(g),\ \Sigma\bigr).
\]

\subsection{Inference rule (MAP / MLE)}
Assume a prior $p(g)$ over goals. The robot infers intent via MAP:
\[
\hat{g}
:= \arg\max_{g\in\mathcal{G}} \ \log p(g) + \sum_{t=0}^{T-1} \log \mathcal{N}\!\bigl(u_t;\mu_t(g),\Sigma\bigr).
\]
Equivalently (dropping constants), define the score
\[
S(g)
:= \log p(g) - \frac12 \sum_{t=0}^{T-1} \bigl(u_t-\mu_t(g)\bigr)^\top \Sigma^{-1}\bigl(u_t-\mu_t(g)\bigr),
\]
and $\hat g = \arg\max_g S(g)$.

\subsection{High-probability correctness guarantee}
Fix a true goal $g^\star\in\mathcal{G}$, and define the pairwise \emph{separation margin} between $g^\star$ and an alternative $g\neq g^\star$:
\[
\Delta_t(g) := \mu_t(g) - \mu_t(g^\star),\qquad
m(g) := \frac12\sum_{t=0}^{T-1} \Delta_t(g)^\top \Sigma^{-1}\Delta_t(g) - \log\frac{p(g)}{p(g^\star)}.
\]
Assume $u_t = \mu_t(g^\star)+\varepsilon_t$ with $\varepsilon_t\sim\mathcal{N}(0,\Sigma)$.
One can show that the MAP decision is correct if for all $g\neq g^\star$,
\[
\sum_{t=0}^{T-1} \Delta_t(g)^\top \Sigma^{-1}\varepsilon_t \ <\ m(g).
\]
The random variable on the left is Gaussian:
\[
Z(g) := \sum_{t=0}^{T-1} \Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
\sim \mathcal{N}\!\Bigl(0,\ V(g)\Bigr),
\quad
V(g) := \sum_{t=0}^{T-1} \Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
\]
Therefore a sufficient condition for pairwise correctness with probability at least $1-\alpha_g$ is
\[
\mathbb{P}\bigl[ Z(g) \ge m(g)\bigr] \le \alpha_g
\quad\Longleftarrow\quad
m(g) \ \ge\ \sqrt{V(g)}\ \Phi^{-1}(1-\alpha_g),
\]
where $\Phi^{-1}$ is the standard normal quantile function.
Using a union bound over $g\neq g^\star$ with $\sum_{g\neq g^\star}\alpha_g \le \alpha$, we obtain:
\[
\mathbb{P}\bigl[\hat g = g^\star\bigr]
\ \ge\ 1-\alpha
\quad\text{if}\quad
\forall g\neq g^\star:\ 
m(g) \ \ge\ \sqrt{V(g)}\ \Phi^{-1}(1-\alpha_g).
\]

\subsection{Robustification to bounded covariance: $\Sigma \preceq \bar{\Sigma}$}
In practice, $\Sigma$ may be unknown but bounded: $\Sigma \preceq \bar{\Sigma}$.
A conservative (workspace-certifiable) guarantee is to require the inequality to hold for the worst-case admissible covariance.
One convenient sufficient condition is to replace $\Sigma^{-1}$ by $\bar{\Sigma}^{-1}$ (since $\Sigma \preceq \bar{\Sigma}\Rightarrow \Sigma^{-1}\succeq \bar{\Sigma}^{-1}$):
\[
\tilde V(g) := \sum_{t=0}^{T-1} \Delta_t(g)^\top \bar{\Sigma}^{-1}\Delta_t(g),
\qquad
\tilde m(g) := \frac12\tilde V(g) - \log\frac{p(g)}{p(g^\star)}.
\]
Then a sufficient, covariance-robust intent-identification condition is:
\[
\forall g\neq g^\star:\ 
\tilde m(g) \ \ge\ \sqrt{\tilde V(g)}\ \Phi^{-1}(1-\alpha_g),
\qquad
\sum_{g\neq g^\star}\alpha_g \le \alpha.
\]
If this holds, then for any $\Sigma\preceq \bar{\Sigma}$,
\[
\mathbb{P}_{\varepsilon_t\sim \mathcal{N}(0,\Sigma)}\bigl[\hat g = g^\star\bigr] \ge 1-\alpha.
\]

\subsection{Workspace design objective / constraint}
Let a workspace design parameter be $\theta\in\Theta$ (object placements, obstacles, etc.) which affects dynamics and thus the predicted means $\mu_t(g;\theta)$ along the interaction:
\[
\mu_t(g;\theta) := \pi_h(x_t(\theta), g; \theta).
\]
Define the robust separations under design $\theta$:
\[
\Delta_t(g;\theta) := \mu_t(g;\theta)-\mu_t(g^\star;\theta),
\quad
\tilde V(g;\theta):=\sum_{t=0}^{T-1}\Delta_t(g;\theta)^\top \bar{\Sigma}^{-1}\Delta_t(g;\theta),
\quad
\tilde m(g;\theta):=\frac12\tilde V(g;\theta)-\log\frac{p(g)}{p(g^\star)}.
\]
Then the \emph{chance-constraint} form of your requirement is:
\[
\text{find }\theta\in\Theta \text{ such that }\ 
\forall \Sigma\preceq \bar{\Sigma}:\ 
\mathbb{P}\bigl[\hat g(\theta)=g^\star\bigr] \ge 1-\alpha,
\]
and a sufficient certifiable constraint is:
\[
\forall g\neq g^\star:\ 
\tilde m(g;\theta) \ \ge\ \sqrt{\tilde V(g;\theta)}\ \Phi^{-1}(1-\alpha_g),
\qquad
\sum_{g\neq g^\star}\alpha_g \le \alpha.
\]
You can then optimize, e.g.,
\[
\max_{\theta\in\Theta}\ \min_{g\neq g^\star}\ 
\Bigl(\tilde m(g;\theta) - \sqrt{\tilde V(g;\theta)}\,\Phi^{-1}(1-\alpha_g)\Bigr)
\quad\text{s.t. feasibility constraints on }\theta.
\]
This maximizes the worst-case ``intent separability slack'' under bounded noise.


\section{Goal switching and $L$-step recovery guarantees under bounded Gaussian input noise}

\subsection{Model}
Let the latent user goal be a discrete-time Markov chain $(g_t)_{t\ge 0}$ on a finite set
$\mathcal{G}=\{1,\dots,M\}$ with transition matrix $A$.
Let the observed joystick input be
\[
u_t = \mu_t(g_t) + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\Sigma),
\qquad \Sigma \preceq \bar{\Sigma},
\]
where $\mu_t(g)\in\mathbb{R}^d$ is the predicted mean input under goal $g$ (given the robot state at time $t$),
and $\bar{\Sigma}\succ 0$ is a known covariance upper bound (Loewner order).

The robot runs a Bayesian filter (HMM filtering) and outputs the MAP estimate
\[
\hat g_t \in \arg\max_{g\in\mathcal{G}} b_t(g),
\qquad
b_t(g) := P(g_t=g\mid u_{0:t}).
\]

\subsection{Switch times and recovery objective}
Define a \emph{switch time} $\tau$ as any time index such that $g_{\tau}\neq g_{\tau-1}$.
We say the filter \emph{recovers within $L$ steps} if
\[
\hat g_{\tau+L} = g_{\tau}
\]
(i.e., after the switch at time $\tau$, the estimator predicts the new goal by time $\tau+L$).

\subsection{A lower bound on posterior mass immediately after switching}
To make any recovery guarantee, we need a lower bound on how much probability the filter assigns
to the newly active goal right after the switch.
Assume there exists $\beta \in (0,1)$ such that at every switch time $\tau$,
\begin{equation}
\label{eq:beta-assumption}
b_{\tau} (g_{\tau}) \;\ge\; \beta.
\end{equation}
This is satisfied, e.g., if the transition model enforces a minimum probability of switching to each goal
(and the filter uses that transition model), but we keep it abstract as \eqref{eq:beta-assumption}.

\subsection{Cumulative separability over an $L$-step window}
Fix a time $t$ and two distinct goals $g\neq h$.
Define the mean-difference and its $\bar{\Sigma}^{-1}$-metric magnitude:
\[
\Delta_s(g,h) := \mu_s(h)-\mu_s(g),\qquad
v_s(g,h) := \Delta_s(g,h)^\top \bar{\Sigma}^{-1}\Delta_s(g,h)\;\ge 0.
\]
Over an $L$-step window starting at $\tau$, define the \emph{cumulative separation}
\[
D_{\tau:L}(g,h) := \sum_{s=\tau}^{\tau+L-1} v_s(g,h).
\]

\subsection{Recovery guarantee (sufficient condition)}
Consider a switch time $\tau$ with new goal $g_\tau$.
Define a worst-case bound on the competing-goal posterior mass at the switch:
\[
b_\tau(h) \le 1-\beta \quad \text{for any } h\neq g_\tau,
\]
so that
\[
\log\frac{b_\tau(h)}{b_\tau(g_\tau)} \le \log\frac{1-\beta}{\beta}.
\]

Let $\alpha\in(0,1)$ be the desired failure probability and set
\[
\alpha' := \frac{\alpha}{M-1}.
\]
A sufficient condition for $L$-step recovery with probability at least $1-\alpha$
(for any $\Sigma\preceq \bar\Sigma$) is:
\begin{equation}
\label{eq:recovery-condition}
\forall h\neq g_\tau:\quad
\frac{1}{2}D_{\tau:L}(g_\tau,h) \;-\; \log\frac{1-\beta}{\beta}
\;\ge\;
\Phi^{-1}(1-\alpha')\,\sqrt{D_{\tau:L}(g_\tau,h)}.
\end{equation}

\paragraph{Theorem (Recovery within $L$ steps).}
Assume \eqref{eq:beta-assumption} holds at every switch time $\tau$.
If the cumulative separability condition \eqref{eq:recovery-condition} holds for that $\tau$,
then
\[
\mathbb{P}\!\left(\hat g_{\tau+L} = g_\tau\right) \;\ge\; 1-\alpha
\qquad\text{for all }\Sigma \preceq \bar{\Sigma}.
\]

\paragraph{Proof sketch (one line).}
For each competitor $h\neq g_\tau$, the log-likelihood ratio over the $L$ observations
is Gaussian with variance $D_{\tau:L}(g_\tau,h)$ and mean $\tfrac12 D_{\tau:L}(g_\tau,h)$ (using $\bar\Sigma^{-1}$
gives a conservative bound for all $\Sigma\preceq \bar\Sigma$); adding the prior term contributes at most
$\log\frac{1-\beta}{\beta}$ against the true goal.
Condition \eqref{eq:recovery-condition} makes the pairwise error probability $\le \alpha'$,
and a union bound over $M-1$ competitors yields total error probability $\le \alpha$.


\subsection{Derivation of the separation margin}

Assume the true goal is $g^\star$, and the joystick input follows
$$
u_t = \mu_t(g^\star) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0,\Sigma).
$$

We use the MAP scoring function
$$
S(g)
=
\log p(g)
-\frac12 \sum_{t=0}^{T-1}
\bigl(u_t-\mu_t(g)\bigr)^\top \Sigma^{-1}\bigl(u_t-\mu_t(g)\bigr).
$$

The estimator predicts
$$
\hat g = \arg\max_{g\in\mathcal G} S(g).
$$

To compare the true goal $g^\star$ against an alternative $g\neq g^\star$, define the score difference
$$
\Delta S(g) := S(g) - S(g^\star).
$$

Substituting the definition of $S(\cdot)$ gives
$$
\Delta S(g)
=
\log\frac{p(g)}{p(g^\star)}
-\frac12 \sum_{t=0}^{T-1}
\Bigl[
\bigl(u_t-\mu_t(g)\bigr)^\top \Sigma^{-1}\bigl(u_t-\mu_t(g)\bigr)
-
\bigl(u_t-\mu_t(g^\star)\bigr)^\top \Sigma^{-1}\bigl(u_t-\mu_t(g^\star)\bigr)
\Bigr].
$$

Now define the mean difference
$$
\Delta_t(g) := \mu_t(g) - \mu_t(g^\star).
$$

Since $u_t = \mu_t(g^\star)+\varepsilon_t$, we have
$$
u_t-\mu_t(g)
=
\mu_t(g^\star)+\varepsilon_t-\mu_t(g)
=
\varepsilon_t-\Delta_t(g),
$$
and
$$
u_t-\mu_t(g^\star)=\varepsilon_t.
$$

Therefore,
$$
\bigl(u_t-\mu_t(g)\bigr)^\top \Sigma^{-1}\bigl(u_t-\mu_t(g)\bigr)
=
(\varepsilon_t-\Delta_t(g))^\top \Sigma^{-1}(\varepsilon_t-\Delta_t(g)).
$$

Expanding the quadratic term,
$$
(\varepsilon_t-\Delta_t(g))^\top \Sigma^{-1}(\varepsilon_t-\Delta_t(g))
=
\varepsilon_t^\top \Sigma^{-1}\varepsilon_t
-2\,\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
+\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
$$

Subtracting the true-goal quadratic gives
$$
(\varepsilon_t-\Delta_t(g))^\top \Sigma^{-1}(\varepsilon_t-\Delta_t(g))
-
\varepsilon_t^\top \Sigma^{-1}\varepsilon_t
=
-2\,\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
+\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
$$

Substituting back into $\Delta S(g)$,
$$
\Delta S(g)
=
\log\frac{p(g)}{p(g^\star)}
-\frac12 \sum_{t=0}^{T-1}
\left(
-2\,\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
+
\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g)
\right).
$$

Distributing the factor of $-\frac12$,
$$
\Delta S(g)
=
\log\frac{p(g)}{p(g^\star)}
+
\sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
-
\frac12 \sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
$$

Rearranging,
$$
\Delta S(g)
=
\underbrace{\sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t}_{Z(g)}
-
\underbrace{\left(
\frac12 \sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g)
-
\log\frac{p(g)}{p(g^\star)}
\right)}_{m(g)}.
$$

Hence we define the separation margin
$$
m(g)
:=
\frac12 \sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g)
-
\log\frac{p(g)}{p(g^\star)}.
$$

With this notation,
$$
\Delta S(g)=Z(g)-m(g).
$$

An error in favor of the alternative goal $g$ occurs when
$$
S(g)\ge S(g^\star)
\quad\Longleftrightarrow\quad
\Delta S(g)\ge 0
\quad\Longleftrightarrow\quad
Z(g)\ge m(g).
$$

Thus $m(g)$ is the deterministic margin that the random noise term $Z(g)$ must overcome in order for the classifier to mistake $g$ for the true goal $g^\star$.

\subsection{Distribution of the noise term}

Since $\varepsilon_t \sim \mathcal{N}(0,\Sigma)$, the scalar random variable
$$
Z(g)
=
\sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\varepsilon_t
$$
is Gaussian with mean
$$
\mathbb{E}[Z(g)] = 0
$$
and variance
$$
\mathrm{Var}[Z(g)]
=
\sum_{t=0}^{T-1}
\Delta_t(g)^\top \Sigma^{-1}\Sigma\Sigma^{-1}\Delta_t(g)
=
\sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
$$

Define
$$
V(g)
:=
\sum_{t=0}^{T-1}\Delta_t(g)^\top \Sigma^{-1}\Delta_t(g).
$$

Then
$$
Z(g)\sim \mathcal{N}(0,V(g)).
$$

Therefore,
$$
\mathbb{P}\bigl(S(g)\ge S(g^\star)\bigr)
=
\mathbb{P}\bigl(Z(g)\ge m(g)\bigr).
$$

This is why $m(g)$ is called the \emph{separation margin}: it quantifies how far apart the true goal $g^\star$ and alternative goal $g$ are in the noise-weighted geometry induced by $\Sigma^{-1}$, adjusted by the prior ratio $\log \frac{p(g)}{p(g^\star)}$.

\end{document}
