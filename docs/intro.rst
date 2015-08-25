Introduction
============

This is a library to assist with calculations for estimation problems using
particle based methods, it contains a number of algorithms such as the
Particle Filter, Auxiliary Particle Filter and support several variants of
Particle Smoothing through the use of Backward Simulation (FFBSi) techniques but
also methods such as the Metropolis-Hastings Backward Proposer (MHBP) and the
Metropolis-Hastings Improved Particle Smoother (MHIPS).

It also provides a framework for doing parameter estimation in nonlinear
models using Expectation Maximization combined with the particle smoothing
algorithms presented above. (PS-EM).

The use of Rao-Blackwellized models is considered an importan special case
and extensive support for it is provided.

The structure is based on presenting a number of interfaces that a problem
specific class must implement in order to use the algorithms. To assisst the
end user base classes for common model structures, such as
Mixed Linear/Nonlinear Gaussian (MLNLG) models are provided to keep the
implementation effort to a minimum.

The idea is to provide an easy prototyping enviroment for testing different
algorithms and model formulations when solving a problem and to act as a
stepping stone for a later more performance oriented problem specific
implementation by the end user. (outside the scope of this framework)

There are three main areas of interest for the end user, where extra focus
has been spent writing clear docstring explaining the expected usage and
behavior

1.
The Simulator class in the pyparticleest.simulator module is the main entry
point for the application, it is used for running the different algorithms
and accessing the results.

2.
The abstract base classes in pyparticleest.interfaces, they define which
operations that needs to be implemented when using different classes of
algorithms.

3.
pyparticleest.models, this modue contains support code for common model
classes, currently:
Nonlinar Gaussian (NLG),
Mixed Linear/Nonlinear Gaussian (MLNLG),
Hierarchical,
Linear Time-Varying (LTV)


You are encourage to study the examples in the documentation, more can be under test/manual in the source code. 
They demonstrate how the framework is instended to be used.

A good starting point is
        test/manual/filtering/basic_model.py

followed by
        test/manual/filtering/nonlin_model.py

For an example of a simple MLNLG model see
        test/manual/smoothing/mlnlg_model.py

and for the commonly used "standard nonlinear model" see
        test/manual/smoothing/standard_nonlin_model.py

