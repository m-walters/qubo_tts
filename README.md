# qubo_tts

Analysis of synthetic quadratic unconstrained binary optimization (QUBO) data, imitating the methods in
[Aramon et al.](https://arxiv.org/pdf/1806.08815.pdf)

Synthetic data was prepared representing the results of four different Monte Carlo algorithms:

- Simulated Annealing (SA)
- Digital Annealing (DA)
- Parallel Tempering with Isoenergetic Clustering Moves (PT+ICM)
- Parallel Tempering Digital Annealing (PTDA)

The DA algorithms are significant because they employ parallelization for the thermalization (MCSweep) steps,
to emulate the Fujitsu device.
