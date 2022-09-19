#!/bin/bash

#just want to remove any run files 
#i.e, delete everything except initial conditions and source code

rm ID*.csv
rm ID*.mat
rm TD*.mat
rm TD*.csv
rm LagrangianStateOut*.csv
rm Conservation*.csv
rm Energy*.csv
rm FHamiltonian*.csv
rm Hamiltonian*.csv
rm -r output_t/
rm -r output_x/
rm ic_input.txt