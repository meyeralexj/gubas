import scipy.integrate as integ
import scipy.io as io
# import matplotlib.pyplot as plots
import numpy as np
import random as rnd
from numpy import linalg as la
from numpy import math as ma

# These functions will be meaningless and incomprehensible without looking over Hou 2016
# all comments are written assuming the reader is familiar with this paper or currently looking at the paper

# u_tilde takes the expansion order (not truncation order) n, the row of matrix tk corresponding to n (input as t), the a and b coefficient sets, 
# the relative position unit vector e, primary inertia integral set TA and rotated secondary inertia integral set TBp - units of TA and TBp should be km, kg
# if you see a bug from indexing here, check the order being input into all functions
# e must be a row vector!!!!
def u_tilde(n,t,a,b,e,TA,TBp):
	n=int(n)
	u=np.zeros([np.size(t),1])# initialize u tilde as 0
	for k in range(n,-1,-2):# loop down from n by 2
		for i1 in range(int(k+1)):# loop through i's based on Hou constraint equation
			for i2 in range(int(k-i1+1)):
				for i3 in range(int(k-i1-i2+1)):
					for i4 in range(int(k-i1-i2-i3+1)):
						for i5 in range(int(k-i1-i2-i3-i4+1)):
							for i6 in [int(k-i1-i2-i3-i4-i5)]:
								for j1 in range(int(n-k+1)):# loop through j's based on Hou constraint equation
									for j2 in range(int(n-k+1-j1)):
										for j3 in range(int(n-k+1-j1-j2)):
											for j4 in range(int(n-k+1-j1-j2-j3)):
												for j5 in range(int(n-k+1-j1-j2-j3-j4)):
													for j6 in [int(n-k-j1-j2-j3-j4-j5)]:
														# do u tilde calculation inside summations	
														u[k//2]+=a[k][i1][i2][i3][i4][i5][i6]\
															*b[n-k][j1][j2][j3][j4][j5][j6]\
															*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
															*e[0,2]**(i3+i6)\
															*TA[i1+j1][i2+j2][i3+j3]\
															*TBp[i4+j4][i5+j5][i6+j6]
		u[k//2]=u[k//2]*t[k//2]#deal with summation over k
	u=sum(u)
	return u

# partial of relative position unit vector with respect to full position
# pass in relative poition unit vector e, relative position magnitude R in km, element of e (de) that partial with respect to element of full position vector (dx)  
#de is index(int) corresponding to de subscript
#dx is index(int) corresponding to dx subscript
def de_dx(e,R,de,dx):
	x=R*e#make full pos vector
	# computation is set up to handle arbitrary elements and return scalar partial, conditionals used to check which form of partial shoukd be calculated
	if de==dx:
		ind=[0,1,2]
		ind.remove(dx)
		val=(x[0,ind[0]]**2+x[0,ind[1]]**2)/(R**3)
	else:
		val=-x[0,de]*x[0,dx]/(R**3)
	return val

## Calculates the partial of u tilde with respect to element dx (integer index) of the full position vector
# inputs: n - mutual potential expansion order (not truncation order)
# t - row n of tk matrix (it is the tk coefficients corresponding to order n
# a,b - a and be coefficient arrays
# e - unit vector of relative position in same frame as TA (must be row oriented)
# R - magnitude of rel pos in km
# dx - element of full rel position (0,1,2) the partial is taken with respect to
# outputs: du_dx_tilde - km kg s unit partial of u tilde with respect to x
def du_dx_tilde(n,t,a,b,e,R,dx,TA,TBp):
	n=int(n)
	de_dx0=de_dx(e,R,0,dx)# get partials of e with respect to dx
	de_dx1=de_dx(e,R,1,dx)
	de_dx2=de_dx(e,R,2,dx)
	du=np.zeros([np.size(t),1])
	for k in range(n,-1,-2):# loop down by 2's from n - see Hou paper
		for i1 in range(int(k+1)):#loop through i's and j's based on constrain in Hou paper
			for i2 in range(int(k-i1+1)):
				for i3 in range(int(k-i1-i2+1)):
					for i4 in range(int(k-i1-i2-i3+1)):
						for i5 in range(int(k-i1-i2-i3-i4+1)):
							for i6 in [int(k-i1-i2-i3-i4-i5)]:
								for j1 in range(int(n-k+1)):
									for j2 in range(int(n-k+1-j1)):
										for j3 in range(int(n-k+1-j1-j2)):
											for j4 in range(int(n-k+1-j1-j2-j3)):
												for j5 in range(int(n-k+1-j1-j2-j3-j4)):
													for j6 in [int(n-k-j1-j2-j3-j4-j5)]:
														if i1+i4==0.:# these conditionals handle how the coefficients ce are set up because certain terms can go to 0 in the partial the way it is written in Hou paper	
															if i2+i5==0.:
																if i3+i6==0.:
																	ce=0.
																else:
																	ce=(i3+i6)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6-1.)*de_dx2
															else:
																if i3+i6==0.:
																	ce=(i2+i5)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5-1.)\
																		*e[0,2]**(i3+i6)*de_dx1
																else:
																	ce=(i3+i6)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6-1.)*de_dx2\
																		+(i2+i5)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5-1.)\
																		*e[0,2]**(i3+i6)*de_dx1
														else:
															if i2+i5==0.:
																if i3+i6==0.:
																	ce=(i1+i4)*e[0,0]**(i1+i4-1.)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6)*de_dx0
																else:
																	ce=(i3+i6)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6-1.)*de_dx2\
																		+(i1+i4)*e[0,0]**(i1+i4-1.)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6)*de_dx0
															else:
																if i3+i6==0.:
																	ce=(i1+i4)*e[0,0]**(i1+i4-1.)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6)*de_dx0\
																		+(i2+i5)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5-1.)\
																		*e[0,2]**(i3+i6)*de_dx1
																else:
																	ce=(i3+i6)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6-1.)*de_dx2\
																		+(i1+i4)*e[0,0]**(i1+i4-1.)*e[0,1]**(i2+i5)\
																		*e[0,2]**(i3+i6)*de_dx0\
																		+(i2+i5)*e[0,0]**(i1+i4)*e[0,1]**(i2+i5-1.)\
																		*e[0,2]**(i3+i6)*de_dx1

														du[k//2]+=a[k][i1][i2][i3][i4][i5][i6]\
															*b[n-k][j1][j2][j3][j4][j5][j6]\
															*TA[i1+j1][i2+j2][i3+j3]\
															*TBp[i4+j4][i5+j5][i6+j6]\
															*ce
		du[k//2]=du[k//2]*t[k//2]# handle tk part of summation in Hou paper
	du=sum(du)												
	return du

## Partial of rotated inertia integral set with respect to rotation matrix element C(i,j)
#inputs: ij - indices of C(i,j) identifying rotation matrix element
#C - rotation matrix
#q - truncation order of inertia integral expansion
#T - inertia integral set being rotated
#ij should be a list of 2 values, the first being i the second being j
def dT_dc(ij,C,q,T):
	dT=np.zeros([q+1,q+1,q+1])
	for l in range(q+1):#loop through inertia integral indices
		for m in range(q+1-l):
			for n in range(q+1-l-m):
				for i1 in range(l+1):# loop through i's and j's as described in Hou equation
					for j1 in range(l-i1+1):
						for i2 in range(m+1):
							for j2 in range(m-i2+1):
								for i3 in range(n+1):
									for j3 in range(n-i3+1):
										if (i1+i2+i3<=q) and (j1+j2+j3<=q)and\
											(l+m+n-i1-i2-i3-j1-j2-j3<=q):# conditional check to ensure constraints are enforced
											c=0. 
											if ij[0]==0:# c is the coefficients in the Hou equation, because some can go to 0 in partial must use conditionals to set up coefficients
												if ij[1]==0 and i1>0:
													c=i1*C[0,0]**(i1-1.)*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
												elif ij[1]==1 and j1>0:
													#print i1,i2,i3,j1,j2,j3,l,m,n
													c=j1*C[0,0]**i1*C[0,1]**(j1-1.)*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
												elif ij[1]==2 and (l-j1-i1)>0:
													c=(l-i1-j1)*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1-1.)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
											elif ij[0]==1:
												if ij[1]==0 and i2>0:
													c=i2*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**(i2-1.)*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
												elif ij[1]==1 and j2>0:
													c=j2*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**(j2-1.)*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
												elif ij[1]==2 and (m-i2-j2)>0:
													c=(m-i2-j2)*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2-1.)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)
											else:
												if ij[1]==0 and i3>0:
													c=i3*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**(i3-1.)*C[2,1]**j3*C[2,2]**(n-i3-j3)
												elif ij[1]==1 and j3>0:
													c=j3*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**(j3-1.)*C[2,2]**(n-i3-j3)
												elif ij[1]==2 and (n-i3-j3)>0:
													c=(n-i3-j3)*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
														*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
														*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3-1.)
											dT[l][m][n]+=(float(ma.factorial(l)/float(ma.factorial(i1)*ma.factorial(j1)\
												*ma.factorial(l-i1-j1))))\
												*(float(ma.factorial(m)/float(ma.factorial(i2)*ma.factorial(j2)\
												*ma.factorial(m-i2-j2))))\
												*(float(ma.factorial(n)/float(ma.factorial(i3)*ma.factorial(j3)\
												*ma.factorial(n-i3-j3))))\
												*c\
												*T[i1+i2+i3][j1+j2+j3][l+m+n-i1-i2-i3-j1-j2-j3]
										# else:
	return dT

## Calculates the partial of u tilde with respect to element C(i,j) of the rotation matrix
# inputs: n - mutual potential expansion order (not truncation order)
# t - row n of tk matrix (it is the tk coefficients corresponding to order n
# a,b - a and be coefficient arrays
# e - unit vector of relative position in same frame as TA (must be row oriented)
# TA - primary set of inertia integrals with units of km and kg
# TBp - rotated secondary inertia integrals with units of km and kg
# dT - partial of rotated secondary inertia integral set with respect to some C(i,j)
# outputs: du_dx_tilde - km kg s unit partial of u tilde with respect to x
def du_dc_tilde(n,t,a,b,e,TA,TBp,dT):
	n=int(n)
	du=np.zeros([np.size(t),1])#0.#
	for k in range(n,-1,-2):# loop down by 2's from n - see Hou paper
		for i1 in range(int(k+1)):# loop through i's and j's base on Hou paper constraint equation
			for i2 in range(int(k-i1+1)):
				for i3 in range(int(k-i1-i2+1)):
					for i4 in range(int(k-i1-i2-i3+1)):
						for i5 in range(int(k-i1-i2-i3-i4+1)):
							for i6 in [int(k-i1-i2-i3-i4-i5)]:
								for j1 in range(int(n-k+1)):
									for j2 in range(int(n-k+1-j1)):
										for j3 in range(int(n-k+1-j1-j2)):
											for j4 in range(int(n-k+1-j1-j2-j3)):
												for j5 in range(int(n-k+1-j1-j2-j3-j4)):
													for j6 in [int(n-k-j1-j2-j3-j4-j5)]:
														du[k/2]+=a[k][i1][i2][i3][i4][i5][i6]\
															*b[n-k][j1][j2][j3][j4][j5][j6]\
															*e[0,0]**(i1+i4)*e[0,1]**(i2+i5)\
															*e[0,2]**(i3+i6)\
															*TA[i1+j1][i2+j2][i3+j3]\
															*dT[i4+j4][i5+j5][i6+j6]
		du[k/2]=du[k/2]*t[k/2]#handles tk part of summation
	du=sum(du)														
	return du

## Computes partial of potential with respect to element of full relative position dx
#inputs: G - gravity constant in units of km kg
#m - mutual potential truncation order
#t - full set of tk coefficients
#a,b - a and b expansion coefficients
#e - unit vector of relative position
#R - magnitude of relative position in km
#TA - primary inertia integrals
#TBp - secondary inertia integrals rotated into A frame
# returns partial of energy potential (negative of force and value in Hou paper)
def du_x(G,m,t,a,b,e,R,dx,TA,TBp):
	m=int(m)
	du=0.
	x=e*R# get full position vec
	for n in range(m+1):# loop from N=0 to truncation order
		du+=(-((n+1.)*(x[0,dx]))/(R**(n+3.)))*u_tilde(n,t[n],a,b,e,TA,TBp)\
			+(1./R**(n+1.))*du_dx_tilde(n,t[n],a,b,e,R,dx,TA,TBp)
	du=-du*G
	return du

## Computes partial of potential with respect to element of rotation matrix C which maps from B to A
#inputs: G - gravity constant in units of km kg
#m - mutual potential truncation order
#t - full set of tk coefficients
#a,b - a and b expansion coefficients
#e - unit vector of relative position
#R - magnitude of relative position in km
#TA - primary inertia integrals
#TBp - secondary inertia integrals rotated into A frame
#dT - partial of secondary inertia integrals rotated into A frame with respect to rotation matrix element C(i,j) where C maps from B to A
# returns partial of energy potential (negative of equation in Hou paper)
def du_c(G,m,t,a,b,e,R,TA,TBp,dT):
	m=int(m)
	du=0.
	for n in range(m+1):# loop from N=0 to truncation order
		du+=(1./(R**(n+1.)))*du_dc_tilde(n,t[n],a,b,e,TA,TBp,dT)
	du=-du*G
	return du

## Computes Energy potential
#inputs: G - gravity constant in units of km kg
#m - mutual potential truncation order
#t - full set of tk coefficients
#a,b - a and b expansion coefficients
#e - unit vector of relative position
#R - magnitude of relative position in km
#TA - primary inertia integrals
#TBp - secondary inertia integrals rotated into A frame
# returns partial of energy potential (negative of force)
def potential(G,m,t,a,b,e,R,TA,TBp):
	m=int(m)
	u=0.
	for n in range(m+1):# loop from N=0 to truncation order
		u+=(1./(R**(n+1.)))*u_tilde(n,t[n],a,b,e,TA,TBp)
	u=-u*G
	return u
