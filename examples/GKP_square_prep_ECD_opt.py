#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


import numpy as np
import qutip as qt 
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer
from ECD_control.ECD_optimization.optimization_analysis import OptimizationAnalysis, OptimizationSweepsAnalysis
from ECD_control.ECD_pulse_construction.ECD_pulse_construction import *
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # Or any other font available on your system
from ECD_control.gate_definitions_qutip import *
from tensorflow.python.client import device_lib


# In[3]:


device_lib.list_local_devices()


# ### A first implementation of GKP States: As eigenstates of finite-energy Pauli operators

# In[8]:


#The target oscillator state.

def gkp_square_states(N, Delta):
    """
    Constructs logical GKP square states in a truncated Hilbert space as the +1 and -1 eigenstates of finite-energy Z_Delta Pauli operator.
    The logical Pauli GKP states can then be obtained by linear combinations of those states. 

    Parameters:
    - N: int, dimension of the Hilbert space (truncation level).
    - Delta: float, squeezing parameter (controls the width of peaks).

    Returns:
    - psi_gkp: Qobj, the computational GKP square states as a QuTiP ket.
    """
   
    # Define infinite-energy Z Pauli operator
    q_op = qt.position(N)
    Z_0 = (1j *np.sqrt(np.pi)*q_op).expm()

    # Define the gaussian enveloppe operator 
    a_op = qt.destroy(N)
    adag_op = a_op.dag()
    E = (-Delta**2 *adag_op * a_op).expm()

    # Define the finite-energy Z Pauli operator 
    Z = E*Z_0*E.inv()
    

    # Determine the GKP codewords as the eigenstates of Z
    eigv, eigs = Z.eigenstates()
    # eigv = eigv.real
    
    # print(eigv)
    # print(eigs)

    psi_plus_Z = eigs[-1] # +1 eigenstate of Z
    psi_minus_Z = eigs[0] # -1 eigenstate of Z

    return (psi_plus_Z.unit(), psi_minus_Z.unit())
    # return ((E*psi_plus_Z_0).unit(), (E*psi_minus_Z_0).unit())



# In[49]:


# Parameters
N = 100  # Hilbert space dimension
Delta = 0.306  # Squeezing parameter


psi_plus_Z, psi_minus_Z = gkp_square_states(N, Delta)
# print(type(psi_plus_Z))
# print(psi_plus_Z.norm())
# print(psi_plus_Z.type)
# print(psi_minus_Z)

xvec=np.linspace(-7.5,7.5,501)
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
ax.grid()
qt.visualization.plot_wigner(psi_plus_Z, xvec, xvec, colorbar=True, fig = fig,ax = ax, g = 2)
plt.savefig("../Figures/Wigner_visualization/psi_plus_Z_N_100_Delta_0.306.svg")
plt.savefig("../Figures/Wigner_visualization/psi_plus_Z_N_100_Delta_0.306.png")


# ### A second implementation: as ground states of the GKP Hamiltonian

# In[4]:


# the target oscillator 

def gkp_square_states(N, omega, Ep, Eq, eta = 1, d = 2):
    """
    Constructs logical GKP square states in a truncated Hilbert space as the ground states of the GKP Hamiltonian.
    The logical Pauli GKP states can then be obtained by linear combinations of those states. 

    Parameters:
    - N: int, dimension of the Hilbert space (truncation level).
    - omega: float, frequency of the harmonic confinement giving rise to the finite-energy Hamiltonian
    - Ep, Eq: float,  energy scales along the directions q and p 
    - eta: float, aspect ratio in phase space. Set to 1 by default for a square lattice
    -d : int, degree of degeneracy for the ground state. Set to 2 by default to get computational GKP states

    Returns:
    - psi_gkp: Qobj, the computation GKP square states as a QuTiP ket.
    """

    # Define momentum and position operator
    p = qt.momentum(N)
    q = qt.position(N)

    # Define the weak harmonic confinement 
    H_harm = (omega/2)*(p**2 + q**2)

    # Define the cosine potentials 
    cos_p_op = ((np.sqrt(2*np.pi*d)/eta)*p).cosm()
    cos_q_op = ((eta*np.sqrt(2*np.pi*d))*q).cosm()

    # Define the finite-energy GKP Hamiltonian
    H = H_harm -Ep*cos_p_op - Eq*cos_q_op

    #  Find the ground states of the GKP Hamiltonian 
    eigv, eigs = H.eigenstates()

    # print(eigv)
    psi_plus_H = eigs[0]
    psi_minus_H = eigs[1]

    return (psi_plus_H,  psi_minus_H)

    


# In[ ]:


N = 100
omega = 1
E = 10

psi_plus_H, psi_minus_H = gkp_square_states(N, omega, E, E)

# print(psi_plus_H.type)
# print(psi_plus_H.norm())


# Construct +Z GKP state from +-H states
psi_plus_Z = np.cos(np.pi/8) * psi_plus_H +np.sin(np.pi/8)*psi_minus_H
# print(psi_plus_Z.norm())
psi_minus_Z = np.sin(np.pi/8)*psi_plus_H - np.cos(np.pi/8)*psi_minus_H

xvec=np.linspace(-5,5,501)
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
ax.grid()
qt.visualization.plot_wigner(psi_minus_Z, xvec, xvec, colorbar=True, fig = fig,ax = ax, g = 2)
# plt.savefig("../Figures/Wigner_visualization/psi_minus_Z_N_100_E_10.svg")
# plt.savefig("../Figures/Wigner_visualization/psi_minus_Z_N_100_E_10.png")


# Define the target state 
psi_t = psi_minus_Z


# In[6]:


#Optimization of ECD Circuit parameters (betas, phis, and thetas)
#the optimization options
opt_params = {
'N_blocks' : 7, #circuit depth
'N_multistart' : 200, #Batch size (number of circuit optimizations to run in parallel)
'epochs' : 200, #number of epochs before termination
'epoch_size' : 10, #number of adam steps per epoch
'learning_rate' : 0.01, #adam learning rate
'term_fid' : 0.995, #terminal fidelitiy
'dfid_stop' : 1e-6, #stop if dfid between two epochs is smaller than this number
'beta_scale' : 3.0, #maximum |beta| for random initialization
'initial_states' : [qt.tensor(qt.basis(2,0),qt.basis(N,0))], #qubit tensor oscillator, start in |g> |0>
'target_states' : [qt.tensor(qt.basis(2,1), psi_t)], #end in |e> |target>.
'name' : 'GKP_square_test', #name for printing and saving
'filename' : None, #if no filename specified, results will be saved in this folder under 'name.h5'
}
#note: optimizer includes pi pulse in every ECD step. However, final ECD step is implemented 
#in experiment as a displacement since the qubit and oscillator should be disentangled at this point.
#So, we ask the optimizer to end in |e> |target> instead of |g>|target>.


# In[7]:


#create optimization object. 
#initial params will be randomized upon creation
opt = BatchOptimizer(**opt_params)

#print optimization info. 
opt.print_info()


# In[ ]:


#run optimizer.
#note the optimizer can be stopped at any time by interrupting the python console,
#and the optimization results will still be saved and part of the opt object.
#This allows you to stop the optimization whenever you want and still use the result.
opt.optimize()


# In[9]:


#can print info, including the best circuit found.
opt.print_info() 


# In[11]:


#can also get the best circuit parameters directly, could save this to a .npz file.
best_circuit =  opt.best_circuit()
betas = best_circuit['betas']
phis = best_circuit['phis']
thetas = best_circuit['thetas']
print(best_circuit)
np.savez('GKP_square_test', betas = betas, phis = phis, thetas = thetas) 

