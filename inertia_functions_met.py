import scipy.integrate as integ
import scipy.io as io
# import matplotlib.pyplot as plots
import numpy as np
import random as rnd
from numpy import linalg as la
from numpy import math as ma

# See Hou 2016 for equation definitions/derivations/descriptions

# computes Q parameter as defined by Hou, where i,j,k are indices for calculation
def Q_ijk(i,j,k):
	q_ijk=float(ma.factorial(i)*ma.factorial(j)*ma.factorial(k))/float(ma.factorial(i+j+k+3.))
	return q_ijk

## computes summation over a tetrahedron defined by 3 coordinates (x,y,z), fourth (x,y,z) is assumed to be at (0,0,0). Computation is based on l,m,n inertia integral order
# l,m,n must be int
# x,y,z's are vertices of tet, assuming fourth is at origin/barycenter
def tet_sums(l,m,n,x1,x2,x3,y1,y2,y3,z1,z2,z3):
	sum_val=0.
	for i1 in range(l+1):# loops through index constraints as defined in Hou
		for j1 in range(l-i1+1):
			for i2 in range(m+1):
				for j2 in range(m-i2+1):
					for i3 in range(n+1):
						for j3 in range(n-i3+1):
							sum_val+=(float(ma.factorial(l))/float(ma.factorial(i1)*ma.factorial(j1)\
								*ma.factorial(l-i1-j1)))\
								*(float(ma.factorial(m))/float(ma.factorial(i2)*ma.factorial(j2)\
								*ma.factorial(m-i2-j2)))\
								*(float(ma.factorial(n))/float(ma.factorial(i3)*ma.factorial(j3)\
								*ma.factorial(n-i3-j3)))\
								*x1**i1*x2**j1*x3**(l-i1-j1)*y1**i2*y2**j2*y3**(m-i2-j2)\
								*z1**i3*z2**j3*z3**(n-i3-j3)\
								*Q_ijk(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3)
	return sum_val

## computes inertia integrals from tet and vert files, files are assumed to define polyhedron in their principle axis of inertia frame
# l,m,n must be int
# rho should be float of density
# tet_file must refer to a csv file with .csv included of tetrahedron defined by 3 vertex numbers
# vert_file must refer to a csv file with .csv included of vertex coords (x,y,z)
#	here we assume vert has 4 columns where first column is index #
#q is truncation order n
def poly_inertia(q,rho,tet_file,vert_file):
	tet=np.genfromtxt(tet_file,delimiter=",",)-1#subtract 1 to match python indexing
	vert=np.genfromtxt(vert_file,delimiter=",")/1000.# convert to km for how this method is implemented
	tet=tet[:,~np.all(np.isnan(tet), axis=0)]
	# print tet
	# print vert
	# print 'end check'
	T=np.zeros([q+1,q+1,q+1])
	for l in range(q+1):# loops through inertia integral l,m,n indices up for truncation order q
		for m in range(q+1-l):
			for n in range(q+1-m-l):
				for a in range(np.shape(tet)[0]):
					x1=vert[int(tet[a,0]),1:4]#get tetrahedron vertices
					x2=vert[int(tet[a,1]),1:4]
					x3=vert[int(tet[a,2]),1:4]
					Ta=np.abs(la.det(np.c_[x1,x2,x3]))# this term is not defined in Hou paper clearly found based on other lit search for similar approaches
					T[l][m][n]+=rho*Ta*tet_sums(l,m,n,x1[0],x2[0],x3[0],x1[1],x2[1],x3[1],x1[2],x2[2],x3[2])# pass inertia integral orders l,m,n and individual coordinates for tet vertices
	return (T)

## this is the same func as above but assuming vertices are passed in km already
# def poly_inertia(l,m,n,rho,tet_file,vert_file):
# 	tet=np.genfromtxt(tet_file,delimiter=",",dtype=int)-1#subtract 1 to match python indexing
# 	vert=np.genfromtxt(vert_file,delimiter=",")
# 	T=0.
# 	for a in range(np.shape(tet)[0]):
# 		x1=vert[tet[a,0],1:4]
# 		x2=vert[tet[a,1],1:4]
# 		x3=vert[tet[a,2],1:4]
# 		Ta=np.abs(la.det(np.c_[x1,x2,x3]))#not certain about need for absolute - comes from test will ellipsoid
# 		#print(Ta)
# 		T+=rho*Ta*tet_sums(l,m,n,x1[0],x2[0],x3[0],x1[1],x2[1],x3[1],x1[2],x2[2],x3[2])
# 	return T

## inertia integral rotation function, takes in a standard rotation matrix, truncation order and principally aligned inertia integral set
# C is rotation matrix
# q is truncation order n
# T in inertia integral set to be rotated
def inertia_rot(C,q,T):
	Tp=np.zeros([q+1,q+1,q+1])
	for l in range(q+1):# loop through inertia integral directional orders l,m,n
		for m in range(q+1-l):
			for n in range(q+1-l-m):
				for i1 in range(l+1):# loop through i's and j's based on Hou summation constraints
					for j1 in range(l-i1+1):
						for i2 in range(m+1):
							for j2 in range(m-i2+1):
								for i3 in range(n+1):
									for j3 in range(n-i3+1):
										if (i1+i2+i3<=q)and(j1+j2+j3<=q)and\
											(l+m+n-i1-i2-i3-j1-j2-j3<=q):# makes sure index constraints are enforced (mostly left in from original dev)
											Tp[l][m][n]+=(float(ma.factorial(l)/float(ma.factorial(i1)*ma.factorial(j1)\
												*ma.factorial(l-i1-j1))))\
												*(float(ma.factorial(m)/float(ma.factorial(i2)*ma.factorial(j2)\
												*ma.factorial(m-i2-j2))))\
												*(float(ma.factorial(n)/float(ma.factorial(i3)*ma.factorial(j3)\
												*ma.factorial(n-i3-j3))))\
												*C[0,0]**i1*C[0,1]**j1*C[0,2]**(l-i1-j1)\
												*C[1,0]**i2*C[1,1]**j2*C[1,2]**(m-i2-j2)\
												*C[2,0]**i3*C[2,1]**j3*C[2,2]**(n-i3-j3)\
												*T[i1+i2+i3][j1+j2+j3][l+m+n-i1-i2-i3-j1-j2-j3]
	return Tp

# polyhedron moment of inertia calculator takes in kg/km3 density, and meter based tet and vert files for polyhedron assuming they are defined in principal frame
# this is not from Hou paper - comes from math paper deriving moments of inertia from point cloud
def poly_moi(rho,tet_file,vert_file):
	tet=np.genfromtxt(tet_file,delimiter=",",)-1#subtract 1 to match python indexing
	vert=np.genfromtxt(vert_file,delimiter=",")/1000.#convert from m to km
	tet=tet[:,~np.all(np.isnan(tet), axis=0)]
	I=np.zeros([1,3])
	M=0.#this is volume included for checking validity in dev
	for a in range(np.shape(tet)[0]):
		p1=np.zeros([1,3])# assumes tet have vertex at 0
		p2=vert[int(tet[a,0]),1:4]# gets other tet vertices
		p3=vert[int(tet[a,1]),1:4]
		p4=vert[int(tet[a,2]),1:4]
		V=rho*np.abs(la.det(np.c_[p2,p3,p4]))/6.#determinant of vertices from lit search
		#print(Ta)
		I[0,0]+=V*(p1[0,1]**2+p1[0,1]*p2[1]+p2[1]**2+p1[0,1]*p3[1]+p2[1]*p3[1]+p3[1]**2+\
			p1[0,2]**2+p1[0,2]*p2[2]+p2[2]**2+p1[0,2]*p3[2]+p2[2]*p3[2]+p3[2]**2+\
			p1[0,1]*p4[1]+p2[1]*p4[1]+p3[1]*p4[1]+p4[1]**2+\
			p1[0,2]*p4[2]+p2[2]*p4[2]+p3[2]*p4[2]+p4[2]**2)/10.# Ixx
		I[0,1]+=V*(p1[0,0]**2+p1[0,0]*p2[0]+p2[0]**2+p1[0,0]*p3[0]+p2[0]*p3[0]+p3[0]**2+\
			p1[0,2]**2+p1[0,2]*p2[2]+p2[2]**2+p1[0,2]*p3[2]+p2[2]*p3[2]+p3[2]**2+\
			p1[0,0]*p4[0]+p2[0]*p4[0]+p3[0]*p4[0]+p4[0]**2+\
			p1[0,2]*p4[2]+p2[2]*p4[2]+p3[2]*p4[2]+p4[2]**2)/10.# Iyy
		I[0,2]+=V*(p1[0,1]**2+p1[0,1]*p2[1]+p2[1]**2+p1[0,1]*p3[1]+p2[1]*p3[1]+p3[1]**2+\
			p1[0,0]**2+p1[0,0]*p2[0]+p2[0]**2+p1[0,0]*p3[0]+p2[0]*p3[0]+p3[0]**2+\
			p1[0,1]*p4[1]+p2[1]*p4[1]+p3[1]*p4[1]+p4[1]**2+\
			p1[0,0]*p4[0]+p2[0]*p4[0]+p3[0]*p4[0]+p4[0]**2)/10.# Izz
		M+=V# volume
	return (I,M)
