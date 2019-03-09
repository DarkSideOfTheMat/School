import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib.style as style
from matplotlib import rc
import seaborn as sns
from math import exp
import operator as op
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D

## Goal is to solve master heat equation and plot the result. ##

def nCr(n, r):
	"""Optimized Choose function
	dependents: from functools import reduce

	returns - (float) result of n!/ r!(n-r)! choose function.
	"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#Fixed values:
delta_t = 0.01
ts = np.arange(0, 6.01, step = delta_t)
d = 0
k = 1.2
nu = 1
ns = np.arange(-25, 26, step = 1)
N = 25



def master_eq(delta, kappa, nu, delta_0 = 0, kappa_0 = .5):
	def w_up(n):
		#Calculate chance of 1 individual switching from group 2 to group 1
		return nu*(N-n)*exp(delta + kappa* (n/N))
	def w_down(n):
		#Calculate chance of 1 individual switching from group 1 to group 2
		return nu*(N+n)*exp(-(delta + kappa* (n/N)))

	def prenormalized(n):
		#Find the initial values
		return nCr(2*N, N + n)*exp(2*delta_0*n + (kappa_0 * n**2)/N)


	C_inverse = sum([prenormalized(n) for n in ns])

	def p_initial(n):
		#Normalize initial values.
		return prenormalized(n)/C_inverse

	def p(n, t):
		#Grab existing probability from solution_df
		return solution_df[t][n]

	def temporal_update(n, t):
		#Should be called on n s.t where n already in solution_df
		
		if n == -N:
			return (w_down(n + 1) * p(n + 1, t) - w_down(n)*p(n, t)) + (-w_up(n)*p(n, t))
		if n == N:
			return (-w_down(n) * p(n - 1, t)) + (w_up(n-1)*p(n-1, t) - w_up(n)*p(n, t))

		return (w_down(n + 1) * p(n + 1, t) - w_down(n)*p(n, t)) + (w_up(n-1)*p(n-1, t)-w_up(n)*p(n, t))

	solution_df = {0: {n : p_initial(n) for n in ns}} #initialize dictionary
	last = 0
	for t in ts: #iteratively solve over time t
		if last or not last and t:
			solution_df[t] = {n : p(n, last) + temporal_update(n, last)*delta_t for n in ns}
			last = t
		
	return pd.DataFrame(solution_df)

df = master_eq(d, k, nu)
#reformatting results
df = df.stack().reset_index().rename({'level_0' :'n', 'level_1' : 't', 0: 'P(n,t)'} ,axis = 'columns')

#Plotting the results using trisurf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(df['t'], df['n'], df['P(n,t)'],cmap=plt.cm.Spectral)

ax.set_xlabel('Time (t)')
ax.set_ylabel('Socio-Config (n)')
ax.set_zlabel('Probability Density P(n,t)')

plt.show()