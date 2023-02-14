This repository contains the benchmark codes for the article: 

Dinkelbach, H.Ü., Vitay, J. and Hamker, F. H. (2019) Scalable simulation of rate-coded and spiking neural networks on shared memory
systems. *2019 Conference on Cognitive Computational Neuroscience*. Berlin (Germany). (available online: https://ccneuro.org/2019/proceedings/0000526.pdf)

In this work we compare several neural simulators

* Auryn: https://fzenke.net/auryn/doku.php
* ANNarchy: https://bitbucket.org/annarchy/annarchy
* Brian2: http://briansimulator.org/
* Brian2GeNN: https://brian2genn.readthedocs.io/en/latest/introduction/
* NEST: https://www.nest-simulator.org/index.html

# rate-coded

This folder contains the implementation of the lin_rate_ipn model. The NEST implementation was based on the documentation of NEST. 

# COBA

This is a repeated benchmark of Vitay et al. 2015 with newer versions of the simulators and extended by Brian2GeNN. We provide the connectivity matrices.

# Literature

Vitay J, Dinkelbach HÜ and Hamker FH (2015). ANNarchy: a code generation approach to neural simulations on parallel hardware. Frontiers in Neuroinformatics 9:19. doi:10.3389/fninf.2015.00019
