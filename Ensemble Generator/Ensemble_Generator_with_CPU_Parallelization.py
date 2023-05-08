'''The code below is a simple implementation of the Wolff Algorithm that is parallelized on multiple CPU cores'''

import numpy as np 
from collections import deque
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
plt.style.use('ggplot')

SAMPLES = 1000  #No of Independant samples that needs to be generated
AUTO = 20  #Autocorrelation time to get independant samples
LOOP = 10 #The same simulation is done LOOP no of times to get SAMPLES*LOOP no of samples

class Cluster3D:
    
    def __init__(self, temp):
        self.temp = temp   #Sampling temperature
        self.width = 20     #Size of the lattice
        self.state = np.ones((self.width,self.width,self.width), dtype=int)   #Creates a 3D lattice with all spins = +1
        self.length = SAMPLES*AUTO
        self.equi = 500    #The model is made to equilibriate before the samples are taken
        self.p_add = 1 - np.exp(-2/temp)     #Probability with which the clusters are formed
        self.auto = 0
        self.avg_size = 0
        
    def neighboring_sites(self, s):
        w = self.width
        return [((s[0]+1)%w, s[1], s[2]), ((s[0]-1)%w, s[1], s[2]),
                 (s[0], (s[1]+1)%w, s[2]), (s[0], (s[1]-1)%w, s[2]),
                 (s[0], s[1], (s[2]+1)%w), (s[0], s[1], (s[2]-1)%w)]
  
    def cluster_flip(self, seed):
        spin = self.state[seed]
        self.state[seed] = -spin  
        cluster_size = 1
        unvisited = deque([seed])   # use a deque to efficiently track the unvisited cluster sites
        while unvisited:   # while unvisited sites remain
            site = unvisited.pop()  # take one and remove from the unvisited list
            for nbr in self.neighboring_sites(site):
                if self.state[nbr] == spin and np.random.random() < self.p_add:
                    self.state[nbr] = -spin
                    unvisited.appendleft(nbr)
                    cluster_size += 1
        return cluster_size
    
    def wolff_cluster_move(self):
        rng = np.random.default_rng()
        seed = tuple(rng.integers(0,self.width,3))
        return self.cluster_flip(seed)
            
    def compute_magnetization(self):
        return np.sum(self.state)/self.width**3

    def compute_internal_energy(self):
        n = self.width
        l = self.state
        e = 0.0
        for i in range(0,n):
            for j in range(0,n):
                for k in range(0,n):
                    if i+1<=n-1:
                        e += -l[i,j,k]*l[i+1,j,k]
                    if j+1<=n-1:
                        e += -l[i,j,k]*l[i,j+1,k]
                    if k+1<=n-1:
                        e += -l[i,j,k]*l[i,j,k+1]
                    if i-1>=0:
                        e += -l[i,j,k]*l[i-1,j,k]
                    if j-1>=0:
                        e += -l[i,j,k]*l[i,j-1,k]
                    if k-1>=0:
                        e += -l[i,j,k]*l[i,j,k-1]
        return e

    def run_ising_wolff_mcmc(self, n):
        total = 0
        for _ in range(n):
            total += self.wolff_cluster_move()
        return total

    def sample_autocovariance(self, x):
        x_shifted = x - np.mean(x)
        return np.array([np.dot(x_shifted[:len(x)-t],x_shifted[t:])/len(x) 
                         for t in range(self.length)])

    def find_correlation_time(self, autocov):
        smaller = np.where(autocov < np.exp(-1)*autocov[0])[0]
        return smaller[0] if len(smaller) > 0 else len(autocov)

    def do_wolff(self):
        trace = np.zeros(self.length)
        energy = np.zeros(self.length)
        total_flips = 0
        self.run_ising_wolff_mcmc(self.equi)
        for i in range(self.length):    
            total_flips += self.wolff_cluster_move()
            trace[i] = self.compute_magnetization()
            energy[i] = self.compute_internal_energy()
            
        autocov = self.sample_autocovariance(np.abs(trace))
        time = self.find_correlation_time(autocov)
        self.avg_size = total_flips/self.length
        self.auto = time
        
        return trace

def generator(temp):
    state = Cluster3D(temp)
    raw_spins = state.do_wolff()
    nets = [raw_spins[-AUTO*i] for i in range(SAMPLES)]
    return abs(np.array(nets))   

if __name__ == '__main__':
    
    u1 = time.time()
    temp = 5.0      #Temperature at which the samples are generated
    
    temps = [temp for i in range(LOOP)]
    raw_spin = []
    pool = Pool()
    data = pool.map(generator, temps)     #Runs the algorithm for the same temperature for LOOP no of times across multiple CPU cores
    raw_spin.append(data)
    spins = np.array(data).flatten()    #Size of spins is SAMPLES*LOOP
   
    u2 = time.time()

    print('Time Taken to Simulate - ', abs(u2-u1))
    
    plt.title('Histogram of the sampled Magnetizations')
    plt.hist(spins, bins=10, label=f'Temp - {temp} K/J')
    plt.grid()
    plt.legend()
    plt.xlabel('Magnetization')
    plt.ylabel('Count')
    
