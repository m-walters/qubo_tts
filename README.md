# qubo_tts

The main goal of [Aramon et al.](https://arxiv.org/pdf/1806.08815.pdf) was to investigate and quantize the advantages
presented by the Fujitsu Digital Annealer CMOS hardware (DA), through the use of Monte Carlo (MC) Ising model simulations.
The Ising model is an appropriate system as it is in the QUBO class of problems, the type of problems the Fujitsu device is
designed to solve.

Four different MC algorithms were evaluated:

- Simulated Annealing (SA)
- Digital Annealing (DA)
- Parallel Tempering with Isoenergetic Clustering Moves (PT+ICM)
- Parallel Tempering Digital Annealing (PTDA)

The DA algorithms are significant because they employ parallelization for the thermalization (MCSweep) steps,
to emulate the Fujitsu device. Additionally, four different coupling interactions were investigated with each algorithm:

- _2D-bimodal_: Two-dimensional, with `{-1, 1}` couplings with equal probability
- _2D-Gaussian_: Two-dimensional, with couplings sampled from a Gaussian distribution
- _SK-bimodal_: Sherrington-Kirkpatrick (SK) problem, a fully connected graph with `{-1,1}` equally selected couplings
- _SK-Gaussian_: SK fully connected graph with Gaussian sampled couplings

## TTS

We are concerned here with re-constructing the Time-to-solution (TTS) calculation algorithm.
This metric is one of the primary values of interest for obvious reasons, and is calculated from the following.

1. We consider a run successful if at some point it reached the "reference energy" during the run. This Bernoulli 
   trial allows for us to represent the probability of $y$ successes in $r$ runs as a bimodal distribution
```math
P(y|\theta, r) = \binom{r}{y}(1-\theta)^{r-y}\theta^{y}
```
where $\theta$ is the success probability for a given run, and is a quantity of interest.

2. For a given setup, we define $R_{99}$ as the number of runs needed to achieve at least one success with a probability of 0.99. Further,
```math
R_{99} = \frac{\log(1-0.99)}{\log(1-\theta)}
```

3. We then determine $TTS = \tau R_{99}$, where $\tau$ is the time to execute a given run. 
$\tau$ was optimized for through a hyperparameter grid search of high/low temperatures and MC sweeps.

The challenge then becomes estimating the probability of success $\theta$.
Instead of inferring this quantity by a point estimate from the fraction of successful runs,
we use Bayesian inference. This allows us to more accurately gauge the variance of different statistics of the TTS.

### Bayesian Inference on $`\theta`$

The appropriate conjugate prior for our parameter $\theta$ is the Jeffreys prior, a $\beta$-distribution with hyperparameters
$\alpha = \beta = 1/2$. The $\beta$ prior is the conjugate for binomial distributions, and the Jeffreys hyperparameters
are selected to make no assumptions about $\theta$.

For a given MC algorithm, interaction coupling, and system size $N$, the group prepared and simulated $I=\{1, 2, ...,
100}\$ initial states, also referred to as "instances".
Each instance was then run $r$ number of times, yielding $y$ successes.
This set of $I$ batches of runs, is resampled with replacement using bootstrapping.
The Bayesian inference algorithm then proceeds as follows:

1. For $B=5000$ bootstrapping iterations, sample from $I$ with replacement
2. An instance, $i$, has a tuple $(r_i, y_i)$ which is used to update our beta prior as $\text{Beta}(\alpha, \beta) 
   \mapsto \text{Beta}(\alpha + y_i, \beta + r_i - y_i)$.
3. We sample an estimate of $\theta$ from this posterior distribution and record this instance of $R_{99,bi}$ (for bootstrap iter $b$).
4. After collecting the set of $R_{99,b}$ values for this bootstrap iter, find the $q$-th percentile and store as $R_{99,bq}$ (of course this is different than $R_{99,bi}$ above).
5. After all bootstrap iterations are complete, we consider the empirical distribution $(\tau R_{99,1q}, ..., \tau R_{99,Bq})$ as an approximation of the true $TTS_q$ distribution.

### Benchmarking

To test our inference algorithm, we put aside the context of Ising simulations and devices and focus on how our algorithm
estimates approach the known values of $\theta$. A given system is characterized by the three hyperparameters $\theta$, $R_{99}$, and $\tau$.
However, $\tau$ is superfluous in the context of this study, so we set it to 1.
In our trials then, we choose a $\theta$ and calculate $R_{99}$ using the logarithmic equation above.

Three investigations are performed where our estimations of known $R_{99}$ values (derived from known $\theta$ values) are evaluated
while varying three parameters independently: $\theta$, $N$, and $B$.
