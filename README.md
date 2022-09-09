# GUBAS
General Use Binary Asteroid Simulator

Welcome to the GUBAS tool, please download the user guide for information on how to set up and use the tool.
The most recent GUBAS files are available in the master branch, and consist of 7 .py files, 1 .cpp file, and 1 .cfg file
The original version of GUBAS is also available in the uploaded zip file. This version contains some minor bugs that have been fixed in the updated files.
The master branch contains GUBAS code built to run on python2. See the relevant branch for a GUBAS version built to run on python3

The goal of the general use binary asteroid simulator is to provide a simple to use, but fast, tool for both observers and mission designers to predict binary asteroid system behaviors. This software tool accomplishes this by implementing the Hou 2016 realization of the full two-body problem (F2BP). The F2BP models binary asteroid systems as two arbitrary mass distributions whose mass elements interact gravitationally and result in both gravity forces and torques. To account for these mass distributions and model the mutual gravity of the F2BP, the inertia integrals of each body are computed up to a user defined expansion order. This approach provides a recursive expression of the mutual gravity potential and represents a significant decrease in the computational burden of the F2BP when compared to other methods of representing the mutual potential.

We ask that you provide the link to this repository and cite the following paper (where it is referenced in Appendix D) if you use the tool in publications:

Alex B. Davis and Daniel J. Scheeres,
Doubly Synchronous Binary Asteroid Mass Parameter Observability,
Icarus, Vol. 341 (2020), https://doi.org/10.1016/j.icarus.2019.113439

If using the 3 body perturbation or solar perturbation functionalities, please also cite the publication:

Alex J. Meyer and Daniel J. Sceeres,
The effect of planetary flybys on singly synchronous binary asteroids,
Icarus, Vol. 367 (2021), https://doi.org/10.1016/j.icarus.2021.114554

-Alex B. Davis & Alex J. Meyer

Errata:
User Guide Eq. 9: The provided definition of the inertia integrals is mass normalized, the code and descriptions in the following equations do not use the mass normalized form
