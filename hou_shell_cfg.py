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
import os
import datetime
import configparser as cfp
import struct 
import csv
from decimal import Decimal

print(datetime.datetime.now())
# np.set_printoptions(precision=20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.20f}".format(x)})
exec(open('./inertia_functions_met.py').read())# inertia integral and tensor calculation and manipulation
exec(open('coefficient_funcs.py').read())# expansion coefficients
exec(open('potential+derivs_func.py').read())# mutual potential and partials functions
exec(open('write_icfile.py').read())# cpp input file writeout
exec(open('benchmark_funcs.py').read())# Fahnestock format readin
exec(open('hou_config_funcs.py').read())# Config file readin

## variables
# read in config file
(G,n,nA,nB,aA,bA,cA,aB,bB,cB,a_shape,b_shape,rhoA,rhoB,t0,tf,tet_fileA,vert_fileA,tet_fileB,vert_fileB,x0,Tgen,integ,h,tol,out_freq,out_time_name,case,flyby_toggle,helio_toggle,sg_toggle,tt_toggle,Mplanet,a_hyp,e_hyp,i_hyp,RAAN_hyp,om_hyp,tau_hyp,Msolar,a_helio,e_helio,i_helio,RAAN_helio,om_helio,tau_helio,sol_rad,au_def,love1,love2,refrad1,refrad2,eps1,eps2,Msun,postProcessing)=hou_config_read("hou_config.cfg")
n=max(nA,nB) # set mutual potential expansion order to max inertia integral expansion order
print("\n### Expansion Order Set To: {val} ###\n".format(val=n))

## compute expansion coefficients for post processing
a=a_calc(n)
b=b_calc(n)
tk=tk_calc(n)

## check to fixed time step and output time step are complimentary
mod_check=np.mod(Decimal(str(out_freq)),Decimal(str(h)))
if out_freq>0 and (out_freq<h or mod_check!=0) and integ!=3:
	raise ValueError('Selection of output frequency compared to integration step size is bad')

## announce integrator selection
if integ ==1:
	print("### Integrator Set to RK4 from {t0} seconds to {tf} seconds with a fixed step of {h} seconds ###".format(t0=t0, tf=tf, h=h))
elif integ ==2:
	print("### Integrator Set to LGVI from {t0} seconds to {tf} seconds with a fixed step of {h} seconds ###".format(t0=t0, tf=tf, h=h))
elif integ ==3:
	print("### Integrtator Set to RK 7(8) from {t0} seconds to {tf} seconds with a tolerance of {tol} ###".format(t0=t0, tf=tf, tol=tol))
	print("+++ This is an adaptive integrator and the output will be post-processed for each step of the integrator +++")
elif integ ==4:
	print("### Integrtator Set to A-B-M from {t0} seconds to {tf} seconds with a fixed step of {h} seconds ###".format(t0=t0, tf=tf, h=h))
else:
	raise ValueError('Invalid Integrator Selection')

# If full shapes are set to be expanded to a lower order than the potential calculations will lose
# the inertia integral set will be too small. If a full shape is being used it makes more sense to 
# treat is accurate enough for arbitrary order
if nA<n and a_shape==2:
	nA=n
	print("### Expansion Order of Primary Shape Increased to Match Integrated Order ###")

if nB<n and b_shape==2:
	nB=n
	print("### Expansion Order of Secondary Shape Increased to Match Integrated Order ###")

## announce body shape selections
if a_shape==0:
	bA=aA
	cA=aA
	print("\n### Primary is a Sphere with Radius: {val} meters ###\n".format(val=aA))
elif a_shape==1:
	print("\n### Primary is an Order {val} Ellipsoid with Semi-Axes: a = {a} meters, b = {b} meters, c = {c} meters ###\n".format(val=nA,a=aA,b=bA,c=cA))
elif a_shape==2:
	print("\n### Primary is a Full Shape Model Computed for Order {val} using Files: {fa} and {fb} ###\n".format(val=nA,fa=tet_fileA, fb=vert_fileA))
else:
	raise ValueError('Bad Shape Selection for Primary')

if b_shape==0:
	bB=aB
	cB=aB
	print("\n### Secondary is a Sphere with Radius: {val} meters ###\n".format(val=aB))
elif b_shape==1:
	print("\n### Secondary is an Order {val} Ellipsoid with Semi-Axes: a = {a} meters, b = {b} meters, c = {c} meters ###\n".format(val=nB,a=aB,b=bB,c=cB))
elif b_shape==2:
	print("\n### Secondary is a Full Shape Model Computed for Order {val} using Files: {fa} and {fb} ###\n".format(val=nB,fa=tet_fileB, fb=vert_fileB))
else:
	raise ValueError('Bad Shape Selection for Secondary')

## write cpp input file
write_icfile(G,n,nA,nB,aA,bA,cA,aB,bB,cB,a_shape,b_shape,rhoA,rhoB,t0,tf,"TDP_"+str(n)+".mat","TDS_"+str(n)+".mat","IDP.mat","IDS.mat",\
	tet_fileA,vert_fileA,tet_fileB,vert_fileB,x0,Tgen,integ,h,tol,flyby_toggle,helio_toggle,sg_toggle,tt_toggle,Mplanet,a_hyp,e_hyp,i_hyp,RAAN_hyp,om_hyp,tau_hyp,Msolar,a_helio,e_helio,i_helio,RAAN_helio,om_helio,tau_helio,sol_rad,au_def,love1,love2,refrad1,refrad2,eps1,eps2,Msun)

## call cpp exe
subprocess.call(["./hou_cpp_final"])

print(datetime.datetime.now())

## convert cpp inertia integral format to python format - 3D matrices use difference index order so must fix this for python
TAc=np.genfromtxt ("TDP_"+str(n)+".csv", delimiter=",")
TBc=np.genfromtxt ("TDS_"+str(n)+".csv", delimiter=",")
if np.shape(TAc)==():
	TAc=np.array([[[TAc]]])
if np.shape(TBc)==():
	TBc=np.array([[[TBc]]])
TAc=TAc.reshape(len(TAc.T),len(TAc.T),len(TAc.T))
TBc=TBc.reshape(len(TBc.T),len(TBc.T),len(TBc.T))
TA=np.zeros(np.shape(TAc))
TB=np.zeros(np.shape(TBc))
for f1 in range(n+1):
	for f2 in range(n+1):
		for f3 in range(n+1):
			TA[f1,f2,f3]=TAc[f3,f1,f2]
			TB[f1,f2,f3]=TBc[f3,f1,f2]

## set up inertias and masses from inertia integrals and inertia tensor files
IA=np.array([np.genfromtxt ('IDP.csv', delimiter=",")])
IB=np.array([np.genfromtxt ('IDS.csv', delimiter=",")])
Mc=TA[0,0,0]
Ms=TB[0,0,0]
m=Mc*Ms/(Mc+Ms)
Ixx_c=IA[0,0]
Iyy_c=IA[0,1]
Izz_c=IA[0,2]
Ixx_s=IB[0,0]
Iyy_s=IB[0,1]
Izz_s=IB[0,2]
mu=G*(Mc+Ms)

## post processing
row_len=30 # xout.bin has row entries of length 30 (30 integrated state variables)
print('Post Processing...')
if not postProcessing:
	print("Just kidding, we are not post processing. Outputing output_t/t_out.bin and output_x/x_out.bin only.")
	print(datetime.datetime.now())
	quit()


if os.path.isfile('LagrangianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
	os.remove('LagrangianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if os.path.isfile('FHamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
	os.remove('FHamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if os.path.isfile('HamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
	os.remove('HamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if os.path.isfile('Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
	os.remove('Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if os.path.isfile('Conservation_Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
	os.remove('Conservation_Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if flyby_toggle==1:
	if os.path.isfile('HyperbolicState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
		os.remove('HyperbolicState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
if helio_toggle==1:
	if os.path.isfile('SolarState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv'):
		os.remove('SolarState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv')
L_f=open('LagrangianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', 'a')
FH_f=open('FHamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', 'a')
H_f=open('HamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', 'a')
EA_f=open('Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', 'a')
CEA_f=open('Conservation_Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', 'a')
if flyby_toggle==1:
	HS_f=open('HyperbolicState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv','a')
if helio_toggle==1:
	Solar_f=open('SolarState_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv','a')

L_w=csv.writer(L_f)
FH_w=csv.writer(FH_f)
H_w=csv.writer(H_f)
EA_w=csv.writer(EA_f)
CEA_w=csv.writer(CEA_f)
if flyby_toggle==1:
	HS_w=csv.writer(HS_f)
if helio_toggle==1:
	Solar_w=csv.writer(Solar_f)

if out_freq==-1 or integ==3: # if a output time file or rk 7(8) are used enter this post processing method
	if integ==3.:
		times=[] 
	else:
		times=list(np.genfromtxt(out_time_name, delimiter=","))
	with open('output_t/t_out.bin', 'rb') as f: # for output time file read through integration output to find specified times
		count=0
		while True:
			t = f.read(8)
			if not t:
				# eof
				break
			t=struct.unpack('d',t)
			if t in times:
					times[times.index(t)]=count
			elif integ==3:
				times.append(count)
			count=count+1
	f.close()
        
	## set up storage for post processing
	# count2=0
	dsize=len(times)
	x_file=open('output_x/x_out.bin',"rb")
	t_file=open('output_t/t_out.bin',"rb")
	kt=np.zeros([1,1])
	kr1=np.zeros([1,1])
	kr2=np.zeros([1,1])
	U=np.zeros([1,1])
	E=np.zeros([1,1])
	H=np.zeros([1,3])
	rpos=np.zeros([1,3])
	lvel=np.zeros([1,3])
	lwc=np.zeros([1,3])
	lws=np.zeros([1,3])
	rmom=np.zeros([1,3])
	cLa=np.zeros([1,3])
	cLb=np.zeros([1,3])
	C_store=np.zeros([1,9])
	Cc_store=np.zeros([1,9])
	fCc_store=np.zeros([1,9])
	tsett=np.zeros([1,1])
	for f in range(count):# count is the number of time steps found in binary during "with open" loop above
		if f in times:# loop through stored lists of output times
			tseek=t_file.seek(f*8)#jump in bits for time binary
			t=struct.unpack('d',t_file.read(8))
			useek=x_file.seek(row_len*f*8)#jump in bits for states binary
			u=struct.unpack(str(row_len)+'d',x_file.read(8*row_len))
			tsett[0]=t#tspan[t]
			## set up states for post processing output and energy/ang mom computation - also converting km kg s values into mks
			cc=np.reshape(u[12:21],[3,3])# rotation matrix from A to N
			c=np.reshape(u[21:30],[3,3])# rotation matrix from B to A
			cs=np.dot(cc,c)# rotation from B to N
			r=np.dot(cc,np.array([u[0:3]]).T)# relative position in N
			rpos[0]=np.array([u[0:3]])*1000.# relative position in A
			vel=np.dot(cc,np.array([u[3:6]]).T)# relative vel in N
			lvel[0]=np.array([u[3:6]])*1000.# relative vel in A
			rmom[0]=m*np.array([u[3:6]])*1000. # relative lin mom
			wc=np.array([u[6:9]]).T# primary ang vel in A frame
			ws=np.dot(c.T,np.array([u[9:12]]).T)# secondary ang vel in B frame
			lwc[0]=u[6:9]# store wc
			lws[0]=np.dot(c.T,np.array([u[9:12]]).T).T[0] # store ws in B frame
			kt[0]=.5*m*np.dot(vel.T,vel)# trans kinetic energy
			kr1[0]=.5*np.dot(wc.T,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))# primary rot kinetic energy
			kr2[0]=.5*np.dot(ws.T,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))# secondary rot kinetic energy
			H[0]=(m*np.cross(r,vel,axis=0)\
			    +np.dot(cc,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))\
			    +np.dot(cs,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))).T#ang mom
			cLa[0]=np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc).T*(1000.**2.)#primary ang mom
			cLb[0]=np.dot(c,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws)).T*(1000.**2.)#secondary ang mom
			C_store[0]=np.reshape(c,[1,9])
			Cc_store[0]=np.reshape(cc,[1,9])
			fCc_store[0]=np.reshape(cc.T,[1,9])
			e=np.dot(cc.T,r/la.norm(r)).T#rel pos unit vector
			R=la.norm(r)#rel pos magnitude
			TBp=inertia_rot(c,n,TB)# rotated secondary inertia integrals
			U[0]=potential(G,n,tk,a,b,e,R,TA,TBp)#mutual potential
			E[0]=U[0]+kt[0]+kr1[0]+kr2[0]#total energy
			if f==0:
				E0=np.copy(E[0])
				H0=np.copy(H[0])
			dE=(E0-E)/E
			dH=(la.norm(H0)-la.norm(H))/la.norm(H0)

			lstateout=np.c_[tsett,rpos,lvel,lwc,lws,C_store,Cc_store,U*(1000.**2)]# lagrangian states set
			fhstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,fCc_store,U*(1000.**2)]# fahnestock formatted hamiltonian states set
			hstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,Cc_store,U*(1000.**2)]# hamiltonian states set
			L_w.writerow(lstateout[0])
			FH_w.writerow(fhstateout[0])
			H_w.writerow(hstateout[0])
			EA_w.writerow(np.c_[E,H][0])
			CEA_w.writerow(np.c_[dE,dH][0])
			# count2=count2+1
else:# if every time step or fixed output frequency do this post processing
	if out_freq==0.:
		out_freq=h# set every time step output frequency based on config file input selection
	seek_var=int(out_freq/h)#jump in binary file between output time steps
	times=np.linspace(t0,tf,int(tf/out_freq+1))# set of time steps to be output
	dsize=len(times)#number of storage entries needed
	x_file=open('output_x/x_out.bin',"rb")
	t_file=open('output_t/t_out.bin',"rb")
	if flyby_toggle==1:
		h_file=open('output_h/h_out.bin',"rb")
	if helio_toggle==1:
		sun_file=open('output_sun/sun_out.bin',"rb")
	##setup storage arrays
	kt=np.zeros([1,1])
	kr1=np.zeros([1,1])
	kr2=np.zeros([1,1])
	U=np.zeros([1,1])
	E=np.zeros([1,1])
	H=np.zeros([1,3])
	rpos=np.zeros([1,3])
	lvel=np.zeros([1,3])
	lwc=np.zeros([1,3])
	lws=np.zeros([1,3])
	temp=np.zeros([1,3])
	temp1=np.zeros([1,3])
	rmom=np.zeros([1,3])
	cLa=np.zeros([1,3])
	cLb=np.zeros([1,3])
	C_store=np.zeros([1,9])
	Cc_store=np.zeros([1,9])
	fCc_store=np.zeros([1,9])
	rhyp=np.zeros([1,3])
	vhyp=np.zeros([1,3])
	rhelio=np.zeros([1,3])
	vhelio=np.zeros([1,3])
	tsett=np.zeros([1,1])
	for f in range(dsize):# loop through number of storage inputs
		tseek=t_file.seek(seek_var*f*8)# jump in bits for time binary
		t=struct.unpack('d',t_file.read(8))
		useek=x_file.seek(row_len*seek_var*f*8)# jump in bits for states binary
		u=struct.unpack(str(row_len)+'d',x_file.read(8*row_len))
		tsett[0]=t#tspan[t]
		if flyby_toggle==1:
			hseek=h_file.seek(6*seek_var*f*8)
			hp=struct.unpack(str(6)+'d',h_file.read(8*6))
		if helio_toggle==1:
			sunseek=sun_file.seek(6*seek_var*f*8)
			solar=struct.unpack(str(6)+'d',sun_file.read(8*6))
		#cc=np.reshape(u[12:21],[3,3]).T#np.reshape(u[12:21],[3,3])
		cc=np.reshape(u[12:21],[3,3])# rotation matrix from A to N
		c=np.reshape(u[21:30],[3,3])# rotation matrix from B to A
		cs=np.dot(cc,c)# rotation matrix from B to N
		r=np.dot(cc,np.array([u[0:3]]).T)# rel pos in N
		rpos[0]=np.array([u[0:3]])*1000.# rel pos in A
		vel=np.dot(cc,np.array([u[3:6]]).T)# rel vel in N
		lvel[0]=np.array([u[3:6]])*1000.# rel vel in A
		rmom[0]=m*np.array([u[3:6]])*1000.# rel mom in A
		wc=np.array([u[6:9]]).T# primary ang vel
		ws=np.dot(c.T,np.array([u[9:12]]).T)# secondary ang vel in B
		temp[0]=np.array([u[9:12]])
		temp1[0]=np.dot(c.T,np.array([u[9:12]]).T).T
		lwc[0]=u[6:9]# store wc
		lws[0]=np.dot(c.T,np.array([u[9:12]]).T).T[0] # store ang vel in B
		kt[0]=.5*m*np.dot(vel.T,vel)# trans kinetic energy
		kr1[0]=.5*np.dot(wc.T,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))#primary rot kinetic energy
		kr2[0]=.5*np.dot(ws.T,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))# secondary rot kinetic energy
		H[0]=(m*np.cross(r,vel,axis=0)\
			+np.dot(cc,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))\
			+np.dot(cs,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))).T#ang mom
		cLa[0]=np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc).T*(1000.**2.)# primary ang mom
		cLb[0]=np.dot(c,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws)).T*(1000.**2.)# secondary ang mom
		C_store[0]=np.reshape(c,[1,9])
		Cc_store[0]=np.reshape(cc,[1,9])
		fCc_store[0]=np.reshape(cc.T,[1,9])
		e=np.dot(cc.T,r/la.norm(r)).T#rel pos unit vector
		R=la.norm(r)# rel pos magnitude
		TBp=inertia_rot(c,n,TB)# rotated secondary inertia integrals
		U[0]=potential(G,n,tk,a,b,e,R,TA,TBp)#mutual potential
		E[0]=U[0]+kt[0]+kr1[0]+kr2[0]# total energy
		if f==0:
			E0=np.copy(E[0])
			H0=np.copy(H[0])
		dE=(E0-E)/E0
		dH=(la.norm(H0)-la.norm(H))/la.norm(H0)
		if flyby_toggle==1:
			rhyp[0]=np.array([hp[0:3]])
			vhyp[0]=np.array([hp[3:6]])
		if helio_toggle==1:
			rhelio[0]=np.array([solar[0:3]])
			vhelio[0]=np.array([solar[3:6]])

		lstateout=np.c_[tsett,rpos,lvel,lwc,lws,C_store,Cc_store,U*(1000.**2)]# lagrangian states set
		fhstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,fCc_store,U*(1000.**2)]# fahnestock formatted hamiltonian states set
		hstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,Cc_store,U*(1000.**2)]# hamiltonian states set
		if flyby_toggle==1:
			hyperbolicout=np.c_[rhyp,vhyp]# 3rd body state set
		if helio_toggle==1:
			solarout=np.c_[rhelio,vhelio]
		L_w.writerow(lstateout[0])
		FH_w.writerow(fhstateout[0])
		H_w.writerow(hstateout[0])
		EA_w.writerow(np.c_[E,H][0])
		CEA_w.writerow(np.c_[dE,dH][0])
		if flyby_toggle==1:
			HS_w.writerow(hyperbolicout[0])
		if helio_toggle==1:
			Solar_w.writerow(solarout[0])

x_file.close()
t_file.close()
if flyby_toggle==1:
	h_file.close()
L_f.close()
FH_f.close()
H_f.close()
EA_f.close()
CEA_f.close()
if flyby_toggle==1:
	HS_f.close()
if helio_toggle==1:
	Solar_f.close()



t_file.close()
x_file.close()
if flyby_toggle==1:
	h_file.close()
if helio_toggle==1:
	sun_file.close()

# if out_freq==-1 or integ==3: # if a output time file or rk 7(8) are used enter this post processing method
# 	if integ==3.:
# 		times=[] 
# 	else:
# 		times=list(np.genfromtxt(out_time_name, delimiter=","))
# 	with open('output_t/t_out.bin', 'rb') as f: # for output time file read through integration output to find specified times
# 		count=0
# 		while True:
# 			t = f.read(8)
# 			if not t:
# 				# eof
# 				break
# 			t=struct.unpack('d',t)
#                         if t in times:
# 				times[times.index(t)]=count
# 			elif integ==3:
# 				times.append(count)
# 			count=count+1
# 	f.close()
        
# 	## set up storage for post processing
# 	count2=0
# 	dsize=len(times)
# 	x_file=open('output_x/x_out.bin',"rb")
#         t_file=open('output_t/t_out.bin',"rb")
#         kt=np.zeros([dsize,1])
#         kr1=np.zeros([dsize,1])
#         kr2=np.zeros([dsize,1])
#         U=np.zeros([dsize,1])
#         E=np.zeros([dsize,1])
#         H=np.zeros([dsize,3])
#         rpos=np.zeros([dsize,3])
#         lvel=np.zeros([dsize,3])
#         lwc=np.zeros([dsize,3])
#         lws=np.zeros([dsize,3])
#         rmom=np.zeros([dsize,3])
#         cLa=np.zeros([dsize,3])
#         cLb=np.zeros([dsize,3])
#         C_store=np.zeros([dsize,9])
#         Cc_store=np.zeros([dsize,9])
#         fCc_store=np.zeros([dsize,9])
#         tsett=np.zeros([dsize,1])
# 	for f in range(count):# count is the number of time steps found in binary during "with open" loop above
# 		if f in times:# loop through stored lists of output times
# 			tseek=t_file.seek(f*8)#jump in bits for time binary
# 			t=struct.unpack('d',t_file.read(8))
# 			useek=x_file.seek(row_len*f*8)#jump in bits for states binary
# 			u=struct.unpack(str(row_len)+'d',x_file.read(8*row_len))
#                 	tsett[count2]=t#tspan[t]
# 			## set up states for post processing output and energy/ang mom computation - also converting km kg s values into mks
#                 	cc=np.reshape(u[12:21],[3,3])# rotation matrix from A to N
#                 	c=np.reshape(u[21:30],[3,3])# rotation matrix from B to A
#                 	cs=np.dot(cc,c)# rotation from B to N
#                		r=np.dot(cc,np.array([u[0:3]]).T)# relative position in N
#                 	rpos[count2]=np.array([u[0:3]])*1000.# relative position in A
#                 	vel=np.dot(cc,np.array([u[3:6]]).T)# relative vel in N
#                         lvel[f]=np.array([u[3:6]])*1000.# relative vel in A
#                 	rmom[count2]=m*np.array([u[3:6]])*1000. # relative lin mom
#                 	wc=np.array([u[6:9]]).T# primary ang vel in A frame
#                 	ws=np.dot(c.T,np.array([u[9:12]]).T)# secondary ang vel in B frame
#                         lwc[f]=u[6:9]# store wc
#                         lws[f]=np.dot(c.T,np.array([u[9:12]]).T).T[0] # store ws in B frame
#                         kt[count2]=.5*m*np.dot(vel.T,vel)# trans kinetic energy
#                 	kr1[count2]=.5*np.dot(wc.T,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))# primary rot kinetic energy
#                 	kr2[count2]=.5*np.dot(ws.T,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))# secondary rot kinetic energy
#                 	H[count2]=(m*np.cross(r,vel,axis=0)\
#                 	        +np.dot(cc,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))\
#                 	        +np.dot(cs,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))).T#ang mom
#                 	cLa[count2]=np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc).T*(1000.**2.)#primary ang mom
#                 	cLb[count2]=np.dot(c,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws)).T*(1000.**2.)#secondary ang mom
#                 	C_store[count2]=np.reshape(c,[1,9])
#                 	Cc_store[count2]=np.reshape(cc,[1,9])
#                 	fCc_store[f]=np.reshape(cc.T,[1,9])
#                 	e=np.dot(cc.T,r/la.norm(r)).T#rel pos unit vector
#                 	R=la.norm(r)#rel pos magnitude
#                 	TBp=inertia_rot(c,n,TB)# rotated secondary inertia integrals
#                 	U[count2]=potential(G,n,tk,a,b,e,R,TA,TBp)#mutual potential
#                 	E[count2]=U[count2]+kt[count2]+kr1[count2]+kr2[count2]#total energy
# 			count2=count2+1
# else:# if every time step or fix output frequency do this post processing
# 	if out_freq==0.:
# 		out_freq=h# set every time step output frequency based on config file input selection
# 	seek_var=int(out_freq/h)#jump in binary file between output time steps
#         times=np.linspace(t0,tf,int(tf/out_freq+1))# set of time steps to be output
# 	dsize=len(times)#number of storage entries needed
# 	x_file=open('output_x/x_out.bin',"rb")
# 	t_file=open('output_t/t_out.bin',"rb")
# 	##setup storage arrays
# 	kt=np.zeros([dsize,1])
# 	kr1=np.zeros([dsize,1])
# 	kr2=np.zeros([dsize,1])
# 	U=np.zeros([dsize,1])
# 	E=np.zeros([dsize,1])
# 	H=np.zeros([dsize,3])
# 	rpos=np.zeros([dsize,3])
# 	lvel=np.zeros([dsize,3])
# 	lwc=np.zeros([dsize,3])
# 	lws=np.zeros([dsize,3])
# 	temp=np.zeros([dsize,3])
# 	temp1=np.zeros([dsize,3])
# 	rmom=np.zeros([dsize,3])
# 	cLa=np.zeros([dsize,3])
# 	cLb=np.zeros([dsize,3])
# 	C_store=np.zeros([dsize,9])
# 	Cc_store=np.zeros([dsize,9])
#         fCc_store=np.zeros([dsize,9])
# 	tsett=np.zeros([dsize,1])
# 	for f in range(dsize):# loop through number of storage inputs
# 		tseek=t_file.seek(seek_var*f*8)# jump in bits for time binary
# 		t=struct.unpack('d',t_file.read(8))
# 		useek=x_file.seek(row_len*seek_var*f*8)# jump in bits for states binary
# 		u=struct.unpack(str(row_len)+'d',x_file.read(8*row_len))
# 		tsett[f]=t#tspan[t]
# 		#cc=np.reshape(u[12:21],[3,3]).T#np.reshape(u[12:21],[3,3])
# 		cc=np.reshape(u[12:21],[3,3])# rotation matrix from A to N
# 		c=np.reshape(u[21:30],[3,3])# rotation matrix from B to A
# 		cs=np.dot(cc,c)# rotation matrix from B to N
# 		r=np.dot(cc,np.array([u[0:3]]).T)# rel pos in N
# 		rpos[f]=np.array([u[0:3]])*1000.# rel pos in A
# 		vel=np.dot(cc,np.array([u[3:6]]).T)# rel vel in N
#                 lvel[f]=np.array([u[3:6]])*1000.# rel vel in A
# 		rmom[f]=m*np.array([u[3:6]])*1000.# rel mom in A
# 		wc=np.array([u[6:9]]).T# primary ang vel
# 		ws=np.dot(c.T,np.array([u[9:12]]).T)# secondary ang vel in B
# 		temp[f]=np.array([u[9:12]])
# 		temp1[f]=np.dot(c.T,np.array([u[9:12]]).T).T
# 		lwc[f]=u[6:9]# store wc
# 		lws[f]=np.dot(c.T,np.array([u[9:12]]).T).T[0] # store ang vel in B
# 		kt[f]=.5*m*np.dot(vel.T,vel)# trans kinetic energy
# 		kr1[f]=.5*np.dot(wc.T,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))#primary rot kinetic energy
# 		kr2[f]=.5*np.dot(ws.T,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))# secondary rot kinetic energy
# 		H[f]=(m*np.cross(r,vel,axis=0)\
# 			+np.dot(cc,np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc))\
# 			+np.dot(cs,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws))).T#ang mom
# 		cLa[f]=np.dot(np.diag([Ixx_c,Iyy_c,Izz_c]),wc).T*(1000.**2.)# primary ang mom
# 		cLb[f]=np.dot(c,np.dot(np.diag([Ixx_s,Iyy_s,Izz_s]),ws)).T*(1000.**2.)# secondary ang mom
# 		C_store[f]=np.reshape(c,[1,9])
# 		Cc_store[f]=np.reshape(cc,[1,9])
#                 fCc_store[f]=np.reshape(cc.T,[1,9])
# 		e=np.dot(cc.T,r/la.norm(r)).T#rel pos unit vector
# 		R=la.norm(r)# rel pos magnitude
# 		TBp=inertia_rot(c,n,TB)# rotated secondary inertia integrals
# 		U[f]=potential(G,n,tk,a,b,e,R,TA,TBp)#mutual potential
# 		E[f]=U[f]+kt[f]+kr1[f]+kr2[f]# total energy
# x_file.close()
# t_file.close()

# dE=[(E[0,0]-E[i,0])/E[0,0] for i in range(len(E))]
# dH=[(la.norm(H[0])-la.norm(H[i]))/la.norm(H[i]) for i in range(len(E))]

# lstateout=np.c_[tsett,rpos,lvel,lwc,lws,C_store,Cc_store,U*(1000.**2)]# lagrangian states set
# fhstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,fCc_store,U*(1000.**2)]# fahnestock formatted hamiltonian states set
# hstateout=np.c_[tsett,rpos,rmom,cLa,cLb,C_store,Cc_store,U*(1000.**2)]# hamiltonian states set
# np.savetxt('LagrangianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', lstateout, delimiter=",")
# np.savetxt('FHamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', fhstateout, delimiter=",")
# np.savetxt('HamiltonianStateOut_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', hstateout, delimiter=",")
# np.savetxt('Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', np.c_[E,H], delimiter=",")
# np.savetxt('Conservation_Energy+AngMom_'+str(tf)+'_'+str(out_freq)+'_'+str(h)+'_'+str(n)+'_'+str(integ)+'_'+case+'.csv', np.c_[dE,dH], delimiter=",")
# t_file.close()
# x_file.close()

print(datetime.datetime.now())
