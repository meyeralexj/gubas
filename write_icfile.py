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

# standard io simply puts variables in a line and moves to next line
def write_icfile(G,n,nA,nB,aA,bA,cA,aB,bB,cB,a_shape,b_shape,rhoA,rhoB,t0,tf,TAfile,TBfile,IAfile,IBfile,tet_fileA,vert_fileA,tet_fileB,vert_fileB,x0,Tgen,integ,h,tol,flybl_toggle,helio_toggle,sg_toggle,tt_toggle,Mplanet,a_hyp,e_hyp,i_hyp,RAAN_hyp,om_hyp,tau_hyp,Msolar,a_helio,e_helio,i_helio,RAAN_helio,om_helio,tau_helio,sol_rad,au_def,love1,love2,refrad1,refrad2,eps1,eps2,Msun):
	file=open("ic_input.txt","w")
	file.write(repr(G))# gravity parameter - should be in units of km and kg
	file.write('\n')
	file.write(repr(n))#mutual potential truncation order
	file.write('\n')
	file.write(repr(nA))#primary shape order truncation
	file.write('\n')
	file.write(repr(nB))#secondary shape order truncation
	file.write('\n')
	file.write(repr(aA))#semi major axis primary in m
	file.write('\n')
	file.write(repr(bA))# semi intermed axis primary in m
	file.write('\n')
	file.write(repr(cA))# semi minor axis primary in m
	file.write('\n')
	file.write(repr(aB))# semi major axis secondary in m
	file.write('\n')
	file.write(repr(bB))# semi intermed axis secondary in m
	file.write('\n')
	file.write(repr(cB))# semi minor axis secondary in m
	file.write('\n')
	file.write(repr(a_shape))# primary shape flag
	file.write('\n')
	file.write(repr(b_shape))# secondary shape flag
	file.write('\n')
	file.write(repr(rhoA))# primary density kg/km3
	file.write('\n')
	file.write(repr(rhoB))# secondary density kg/km3
	file.write('\n')
	file.write(repr(t0))# initial time seconds
	file.write('\n')
	file.write(repr(tf))# final time seconds
	file.write('\n')
	file.write(TAfile)# primary inertia integral set file name - functionality not fully implemented - inertia integral computation is very quick no need to avoid recomputing
	file.write('\n')
	file.write(TBfile)# secondary inertia integral set file name - functionality not fully implemented
	file.write('\n')
	file.write(IAfile)# primary inertia moments set file name - functionality not fully implemented
	file.write('\n')
	file.write(IBfile)# secondary inertia moments set file name - functionality not fully implemented
	file.write('\n')
	file.write(tet_fileA)# primary tet file
	file.write('\n')
	file.write(vert_fileA)# primary vert file
	file.write('\n')
	file.write(tet_fileB)# secondary tet file
	file.write('\n')
	file.write(vert_fileB)# secondary ver file
	file.write('\n')
	for i in range(30):# step through states - units of km, s and rad where applicable - [r, v, wc, ws, Cc, C]
		file.write(repr(x0[i]))
		file.write('\n')
	file.write(repr(Tgen))# inertia integral generation flag
	file.write('\n')
	file.write(repr(integ))# inegrator choice flag
	file.write('\n')
	file.write(repr(h))# fixed step time - seconds
	file.write('\n')
	file.write(repr(tol))# adaptive tolerance for rk 78
	file.write('\n')
	file.write(repr(flyby_toggle))
	file.write('\n')
	file.write(repr(helio_toggle))
	file.write('\n')
	file.write(repr(sg_toggle))
	file.write('\n')
	file.write(repr(tt_toggle))
	file.write('\n')
	file.write(repr(Mplanet))
	file.write('\n')
	file.write(repr(a_hyp))
	file.write('\n')
	file.write(repr(e_hyp))
	file.write('\n')
	file.write(repr(i_hyp))
	file.write('\n')
	file.write(repr(RAAN_hyp))
	file.write('\n')
	file.write(repr(om_hyp))
	file.write('\n')
	file.write(repr(tau_hyp))
	file.write('\n')
	file.write(repr(Msolar))
	file.write('\n')
	file.write(repr(a_helio))
	file.write('\n')
	file.write(repr(e_helio))
	file.write('\n')
	file.write(repr(i_helio))
	file.write('\n')
	file.write(repr(RAAN_helio))
	file.write('\n')
	file.write(repr(om_helio))
	file.write('\n')
	file.write(repr(tau_helio))
	file.write('\n')
	file.write(repr(sol_rad))
	file.write('\n')
	file.write(repr(au_def))
	file.write('\n')
	file.write(repr(love1))
	file.write('\n')
	file.write(repr(love2))
	file.write('\n')
	file.write(repr(refrad1))
	file.write('\n')
	file.write(repr(refrad2))
	file.write('\n')
	file.write(repr(eps1))
	file.write('\n')
	file.write(repr(eps2))
	file.write('\n')
	file.write(repr(Msun))
	file.write('\n')
	return
