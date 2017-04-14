=========================================================
Asynchronous Particle swarm optimization (PSO) with constraint support
=========================================================

This is a fork of Pyswarm 0.7 by tisimst.  

This fork adds parallel asynchronous particle support, as described here:

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1769316/

The key difference is that particles do not wait until the end of an iteration to start the next run or update the global best.  This prevents CPUs from idling when some function evaluations take significantly longer than others.

Intended for inversion of computationally expensive functions using many parallel evaluations.  I (intend to) use it to match free parameters in FEM simulations.

Current Status
==============

Works, but probably buggy and need further testing.  Not ready for use.

Requirements
============

- NumPy

Installation and download
=========================

This is intended as a drop-in replacement for pyswarm.  The only files changed is this readme, and pso.py.

To have it mimic the behavior of pyswarm 0.7 exactly, pass an extra argument async=False when calling pso.


License
=======

This package is provided under two licenses:

1. The *BSD License*
2. Any other that the author approves (just ask!)
