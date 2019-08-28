# GUBAS
General Use Binary Asteroid Simulator

Welcome the the GUBAS tool, please download the user guide for information on how to set up and use the tool.
All necessary files can be found in the uploaded zip file.

The goal of the general use binary asteroid simulator is to provide a simple to use, but fast, tool for both observers and mission designers to predict binary asteroid system behaviors. This software tool accomplishes this by implementing the Hou 2016 realization of the full two-body problem (F2BP). The F2BP models binary asteroid systems as two arbitrary mass distributions whose mass elements interact gravitationally and result in both gravity forces and torques. To account for these mass distributions and model the mutual gravity of the F2BP, the inertia integrals of each body are computed up to a user defined expansion order. This approach provides a recursive expression of the mutual gravity potential and represents a significant decrease in the computational burden of the F2BP when compared to other methods of representing the mutual potential.

I ask that you provide the link to this repository and cite the following paper (where it is referenced in Appendix D) if you use the tool in publications:

Alex B. Davis and Daniel J. Scheeres,
Doubly Synchronous Binary Asteroid Mass Parameter Observability,
Icarus, In Review

-Alex B. Davis
