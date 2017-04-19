=================================================================================
Parallel Asynchronous Particle Swarm Optimization (PAPSO) With Constraint Support
=================================================================================

This is a fork of Pyswarm 0.7 by tisimst (https://github.com/tisimst/pyswarm).  This fork adds parallel asynchronous particle support, as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1769316/.

The key difference is that particles do not wait until the end of an iteration to start the next run or update the global best.  This prevents CPUs from idling when some function evaluations take significantly longer than others.  Intended for inversion of computationally expensive functions using many parallel evaluations.  I use it to match free parameters in FEM simulations, and to tune pyswarm itself using a reduced-order model.

Current Status
==============

Works far as I can tell, but need further testing.  Use at own risk.

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