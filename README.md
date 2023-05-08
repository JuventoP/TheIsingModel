# The Ising Model
The Ising Model is a simple model of a magnet. It is a 3D lattice of size NxNxN with each site taking a spin value of +1 or -1. The Hamiltonian for this system is given by, 

$$\mathcal{H} = -H\sum_{i} \sigma_i - J\sum_{i,j} \sigma_i \sigma_j $$ 

and the Partition function is given by,

$$\mathcal{Z} = tr\(e^{-\beta \mathcal{H}})$$ 
where $\beta$ is the inverse of the temperature

There are several algorithms that can simulate the Ising Model as per the given dynamics. One such method is the Wolff Algorithm. It is a Markov Chain Monte Carlo technique, that chooses a random seed, forms cluster of similar spins by considering its nearest neighbors and their nearest neighbors and so on till the entire lattice is covered. This cluster is then flipped all together at once and that constitutes as one Wolff move. 

This document includes the explicit working of the Wolff Algorithm and how it is used to generate ensembles of independant samples of spin configs. Also includes how the implementation can be paralleleized across CPUs for computational speedup. The primary structure of the code is taken from https://hef.ru.nl/~tbudd/mct/lectures/cluster_algorithms.html
