import scipy.integrate as integ
import scipy.io as io
# import matplotlib.pyplot as plots
import numpy as np
import random as rnd
from numpy import linalg as la
# from numpy import math as ma #this import no longer necessary - harrison
import math #we should use the standard math library instead of numpy.math

# Hou expansion coefficients - see Hou 2016 for equation definitions/descriptions

#generate tk expansion coefficients where m is mutual potential truncation order - output tk is matrix with rows of expansion order and columns of recursion steps
def tk_calc(m):
	t=np.zeros([m+1,m//2+2])# create t matrix
	for n in range(m+1):# loop through expansion orders up to truncation order
		#in the following math.factorial() functions, it expects an integer but n/2. gives a float
		#I suspect that this didn't trigger an error in python2
		#but it errors with my python install. 
		#for now I just use int(), which should be appropriate... -harrison
		#also im not sure what/why so many float() functiosn are needed but leaving it for now
		if np.mod(n,2.):# if odd
			t[n,0]=(-1.)**((n-1.)/2.)*float(math.factorial(n))/(2.**(n-1.)\
				*(float(math.factorial(int((n-1.)/2.)))**2))
		else:# if even
			t[n,0]=(-1.)**(n/2.)*float(math.factorial(n))/(2.**n\
				*(float(math.factorial(int(n/2.)))**2))
		k=np.mod(n,2.)#set k looping index
		i=1# set i looping index
		while k<=n:# recursion loop
			t[n,i]=-(n-k)*(n+k+1.)*t[n,i-1]/((k+2.)*(k+1.))
			# print k,t[n],t[n,i-1]
			k+=2.
			i+=1
	return t

#generate a expansion coefficents where n is mutual pptential truncation order - output is a 7 dimensional array
def a_calc(n):
	n=int(n)
	a=np.zeros([n+1,n+1,n+1,n+1,n+1,n+1,n+1])#7 dimensional, 1st dim is k, others ar i
	a[0][0][0][0][0][0][0]=1.# set initial a values
	if n>0:
		a[1][1][0][0][0][0][0]=1.# set a order 1 values as defined in Hou paper
		a[1][0][1][0][0][0][0]=1.
		a[1][0][0][1][0][0][0]=1.
		a[1][0][0][0][1][0][0]=-1.
		a[1][0][0][0][0][1][0]=-1.
		a[1][0][0][0][0][0][1]=-1.
		if n>1:
			for k in range(2,int(n+1)):# recursion loop, looping of i's is based on constraint equation for i's defined in Hou paper
				for i1 in range(int(k+1)):
					for i2 in range(int(k-i1+1)):
						for i3 in range(int(k-i1-i2+1)):
							for i4 in range(int(k-i1-i2-i3+1)):
								for i5 in range(int(k-i1-i2-i3-i4+1)):
									for i6 in range(int(k-i1-i2-i3-i4-i5+1)):
										if i1 >0:
											a[k][i1][i2][i3][i4][i5][i6]+=a[k-1][i1-1][i2][i3][i4][i5][i6]
										if i2 >0:
											a[k][i1][i2][i3][i4][i5][i6]+=a[k-1][i1][i2-1][i3][i4][i5][i6]
										if i3 >0:
											a[k][i1][i2][i3][i4][i5][i6]+=a[k-1][i1][i2][i3-1][i4][i5][i6]
										if i4 >0:
											a[k][i1][i2][i3][i4][i5][i6]-=a[k-1][i1][i2][i3][i4-1][i5][i6]
										if i5 >0:
											a[k][i1][i2][i3][i4][i5][i6]-=a[k-1][i1][i2][i3][i4][i5-1][i6]
										if i6 >0:
											a[k][i1][i2][i3][i4][i5][i6]-=a[k-1][i1][i2][i3][i4][i5][i6-1]
	return a

#generate b expansion coefficents where n is mutual pptential truncation order - output is a 7 dimensional array
def b_calc(n):
	n=int(n)
	b=np.zeros([n+1,n+1,n+1,n+1,n+1,n+1,n+1])#7 dimensional, 1st dim is k, others are i
	b[0][0][0][0][0][0][0]=1.# set initial b values
	# print n
	if n>1:
		b[2][2][0][0][0][0][0]=1.# set order 2 values as defined in Hou paper
		b[2][0][2][0][0][0][0]=1.
		b[2][0][0][2][0][0][0]=1.
		b[2][0][0][0][2][0][0]=1.
		b[2][0][0][0][0][2][0]=1.
		b[2][0][0][0][0][0][2]=1.
		b[2][1][0][0][1][0][0]=-2.
		b[2][0][1][0][0][1][0]=-2.
		b[2][0][0][1][0][0][1]=-2.
		# print n
		for k in range(n,-1,-1):# recursion loop, looping of j's is based on constraint equation in Hou paper
			# print k
			for j1 in range(int(n-k+1)):
				for j2 in range(int(n-k+1-j1)):
					for j3 in range(int(n-k+1-j1-j2)):
						for j4 in range(int(n-k+1-j1-j2-j3)):
							for j5 in range(int(n-k+1-j1-j2-j3-j4)):
								for j6 in range(int(n-k+1-j1-j2-j3-j4-j5)):
									# print 'a'
									if n-k>2:
										# print 'b'
										if j1>0 and j4>0:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												-2.*b[n-k-2][j1-1][j2][j3][j4-1][j5][j6]
										if j2>0 and j5>0:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												-2.*b[n-k-2][j1][j2-1][j3][j4][j5-1][j6]
										if j3>0 and j6>0:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												-2.*b[n-k-2][j1][j2][j3-1][j4][j5][j6-1]
										if j1>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1-2][j2][j3][j4][j5][j6]
										if j2>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1][j2-2][j3][j4][j5][j6]
										if j3>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1][j2][j3-2][j4][j5][j6]
										if j4>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1][j2][j3][j4-2][j5][j6]
										if j5>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1][j2][j3][j4][j5-2][j6]
										if j6>1:
											b[n-k][j1][j2][j3][j4][j5][j6]+=\
												b[n-k-2][j1][j2][j3][j4][j5][j6-2]
	return b
