=========================================================
Parallel Asynchronous Particle Swarm Optimization (PAPSO)
=========================================================

This is a fork of Pyswarm 0.7 by tisimst. This fork adds parallel asynchronous particle support, as well as a few other improvements.

The key difference is that particles do not wait until the end of an iteration to start the next run or update the global best.  This prevents CPUs from idling when some function evaluations take significantly longer than others.  Intended for inversion of computationally expensive functions using many parallel evaluations.  I use it to match free parameters in FEM simulations, and to tune pyswarm itself using a reduced-order model.

Current Status
==============

Works far as I can tell, but need further testing.  Use at own risk.

Example
=======

.. code:: python

	from pyswarm import pso
	
	xopt, fopt = pso(sim, lb, ub, f_ieqcons=constraints, swarmsize=10, omega=0.5, phip=0.5, phig=5, maxiter=100, minstep=0.0001, minfunc=0.000001, processes=10, debug=True, async=True, particle_output=False, initial_best_guess=initial_best_guess, quiet=False)
	
	# alternatively:
	
	xopt, fopt, xlog, fxlog = pso(sim, lb, ub, f_ieqcons=constraints, swarmsize=10, omega=0.5, phip=0.5, phig=5, maxiter=100, minstep=0.0001, minfunc=0.000001, processes=10, debug=True, async=True, particle_output=True, initial_best_guess=initial_best_guess, quiet=False)

Input
=====
	
sim 
  the function to be optimized.  It takes as input a list x.  Returns a real number representing the cost.
lb,ub 
  are the upper/lower bounds.  len(lb) == len(ub) == len(x)
f_ieqcons 
  takes as input a list x.  Returns 1 for acceptable values of x and 0 for unacceptable values of x.
swarmsize 
  is the number of particles.
omega, phip, phig
  PSO parameters.  Need to be tuned to the problem.
maxiter 
  number of iterations to run.  For the asynchronous version, it is approximately total runs / number of particles.
minstep 
  minimum euclidean distance between two consecutive bests before the function quits.
minfunc 
  minimum cost difference between two consecutive bests before the function quits.
processes 
  how many threads to use.  Unlike the upstream Pyswarm which uses the multiprocessing library, this code uses multiprocessing.dummy.
debug 
  prints some extra info to screen.
async 
  this enables asynchronous particles.  See description above.  Defaults to True.  async=False mimics Pyswarm 0.7 behavior.
particle_output 
  enables output of xlog and fxlog; see below.
initial_best_guess 
  a list containing an initial position and cost.  Speeds convergence if you have it, but is optional.  e.g. [[10, 2],0.2]
quiet 
  disables all screen output.
	
Output
======

xopt 
  position of optimized result.
fopt 
  cost of optimized result.
xlog 
  a dictionary containing evaluated positions for each particle.
fxlog 
  a dictionary containing the cost for each evaluation for each particle.
	
	
Requirements
============

- NumPy

Installation and download
=========================

This is a drop-in replacement for pyswarm.  The only files changed is this readme, pso.py, and setup.py.  If you use particle_output=True, returned object has slightly different format (they are dictionaries).

To have it mimic the behavior of pyswarm 0.7 exactly, pass an extra argument async=False when calling pso.

There is also a new quiet=True flag that disables all screen output.


License
=======

This package is provided under two licenses:

1. The *BSD License*
2. Any other that the author approves (just ask!)