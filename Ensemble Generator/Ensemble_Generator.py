'''The code below is a basic implementation of the Wolff Algorithm. Quantities like Net Magnetization, Internal Energy, Average Cluster Size, Autocorrelation Time are calculated'''

import numpy as np 
from collections import deque
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

SAMPLES = 10000  #No of Independant samples that needs to be generated
AUTO = 50  #Autocorrelation time to get independant samples

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
        
        return trace, energy, self.avg_size, self.auto

def generator(temp):
    state = Cluster3D(temp)
    raw_spins, raw_energy, avg_size, auto = state.do_wolff()
    nets = [raw_spins[-AUTO*i] for i in range(SAMPLES)]
    energy = [raw_energy[-AUTO*i] for i in range(SAMPLES)]
    return abs(np.array(nets)), abs(np.array(energy)), avg_size, auto      

if __name__ == '__main__':
    
    u1 = time.time()
    temp = 5.0      #Temperature at which the samples are generated
    spins, energy, avg_size, auto = generator(temp)    #spins and energy are arrays of independant values of net magnetization and internal energy for a given temperature
    u2 = time.time()

    print('Time Taken to Simulate - ', abs(u2-u1), 'Autocorrelation Time - ', auto, 'Average Cluster Size - ', avg_size)

    '''Below lies the histogram plots of the sampled values of Net Magnetization and Internal Energy'''
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    
    ax[0].set_title('Histogram of the sampled Magnetizations')
    ax[0].hist(spins, bins=10, label=f'Temp - {temp} K/J')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('Magnetization')
    ax[0].set_ylabel('Count')
    
    ax[1].set_title('Histogram of the sampled Internal Energies')
    ax[1].hist(energy, bins=10, label=f'Temp - {temp} K/J')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('Internal Energy')
    ax[1].set_ylabel('Count')
 
