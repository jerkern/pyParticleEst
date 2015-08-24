Interfaces
==========

Particle Filter
###############
.. autoclass:: pyparticleest.interfaces.SIR
        :members:

.. autoclass:: pyparticleest.interfaces.ParticleFiltering
        :members:

.. autoclass:: pyparticleest.interfaces.FFProposeFromMeasure
        :members:

Particle Filter Non-Markov
##########################
.. autoclass:: pyparticleest.interfaces.ParticleFilteringNonMarkov
        :members:

Auxiliary Particle Filter
#########################
.. autoclass:: pyparticleest.interfaces.AuxiliaryParticleFiltering
        :members:

Forward-Filter Backward Simulator
#################################
.. autoclass:: pyparticleest.interfaces.FFBSi
        :members:

FFBSi Non-Markov
#################################
.. autoclass:: pyparticleest.interfaces.FFBSiNonMarkov
        :members:


FFBSi with Rejection Sampling
#############################
.. autoclass:: pyparticleest.interfaces.FFBSiRS
        :members:

.. autoclass:: pyparticleest.interfaces.FFBSiRSNonMarkov
        :members:


Proposal methods, e.g. MHIPS and MHBP
#####################################
.. autoclass:: pyparticleest.interfaces.SampleProposer
        :members:

Parameter estimation
####################

Numerical gradient
******************
.. autoclass:: pyparticleest.paramest.interfaces.ParamEstInterface
        :members:

Analytic gradient
*****************
.. autoclass:: pyparticleest.paramest.interfaces.ParamEstInterface_GradientSearch
        :members:

