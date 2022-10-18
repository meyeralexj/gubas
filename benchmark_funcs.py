import scipy.integrate as integ
import scipy.io as io
# import matplotlib.pyplot as plots
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.tri as mtri
# from matplotlib import animation
import numpy as np
import random as rnd
from numpy import linalg as la
from numpy import math as ma
import subprocess
import datetime

def read_bench(filep,files):# filep is parameters file, files is states file both in Fahnestock format
	# fin=open(filep,'r')#load parameters files
	# params = fin.readline()
	# params = map(float, params.strip().split('  '))
	# fin.close()
	params = np.genfromtxt(filep) #added by hagrusa

	rhoA=params[0]*1000.**3#get density - convert from kg/m3 to kg/km3
	rhoB=params[1]*1000.**3
	IA=np.array(params[4:13])# get inertias
	IB=np.array(params[13:22])
	IA=np.reshape(IA,[3,3])
	IB=np.reshape(IB,[3,3])
	Mc=params[22]#primary mass
	Ms=params[23]#secondary mass
	m=params[24]#mass ratio
	G=params[31]/(1000.**3)#gravity constant

	# fin=open(files,'r')
	# states = fin.readline()#load states files
	# states = map(float, states.strip().split('  '))
	# fin.close()
	states = np.genfromtxt(files) #added by hagrusa
	
	r0=np.array(states[0:3])/1000.#rel pos in A - m to km
	v0=np.array(states[3:6])/m/1000.# rel vel in A - m to km
	wc0=np.dot(la.inv(IA),np.array([states[6:9]]).T)# get primary ang vel in A
	C0=states[12:21]# get rotation from B to A - this is wrapped by row
	Cc0=states[21:30]# get rotation from A to N - this is wrapped by column so in x0 below must reorder to wrapping by column
	ws0=np.dot(np.reshape(np.array([C0]),[3,3]),np.dot(la.inv(IB),np.dot(np.reshape(np.array([C0]),[3,3]).T,np.array([states[9:12]]).T)))
	# ws0=np.dot(np.reshape(np.array([C0]),[3,3]),np.array([params[28:31]]).T)# get ang vel in A and convert to B

	x0=[r0[0],r0[1],r0[2],v0[0],v0[1],v0[2],wc0[0,0],wc0[1,0],wc0[2,0],ws0[0,0],ws0[1,0],ws0[2,0],\
	Cc0[0],Cc0[3],Cc0[6],Cc0[1],Cc0[4],Cc0[7],Cc0[2],Cc0[5],Cc0[8],\
	C0[0],C0[1],C0[2],C0[3],C0[4],C0[5],C0[6],C0[7],C0[8]]# order states into this method's state ordering
	return(G,rhoA,rhoB,x0)
