# GUBAS
General Use Binary Asteroid Simulator - *You are currently looking at a modified version includes optional tidal torque and solar gravity effects

Welcome the the GUBAS tool, please download the user guide for information on how to set up and use the tool.
All necessary files can be found in the uploaded zip file.

The goal of the general use binary asteroid simulator is to provide a simple to use, but fast, tool for both observers and mission designers to predict binary asteroid system behaviors. This software tool accomplishes this by implementing the Hou 2016 realization of the full two-body problem (F2BP). The F2BP models binary asteroid systems as two arbitrary mass distributions whose mass elements interact gravitationally and result in both gravity forces and torques. To account for these mass distributions and model the mutual gravity of the F2BP, the inertia integrals of each body are computed up to a user defined expansion order. This approach provides a recursive expression of the mutual gravity potential and represents a significant decrease in the computational burden of the F2BP when compared to other methods of representing the mutual potential.

*For this version of the code which includes perturbations, the C++ script/executable and Python config file document the use of these capabilities in detail. The structure remain functionally identical to the version described in the user guide.

I ask that you provide the link to this repository and cite the following paper (where it is referenced in Appendix D) if you use the tool in publications:

Alex B. Davis and Daniel J. Scheeres,
Doubly Synchronous Binary Asteroid Mass Parameter Observability,
Icarus, Vol. 341 (2020), https://doi.org/10.1016/j.icarus.2019.113439

When using the perturbation models we request that you cite the following paper, which describes the perterbation models implemented in teh code:

Alex B. Davis and Daniel J. Scheeres,
High Fidelity Modeling of Rotationally Fissioned Asteroids,
In Review

-Alex B. Davis

Errata:
User Guide Eq. 9: The provided definition of the inertia integrals is mass normalized, the code and descriptions in the following equations do not use the mass normalized form
