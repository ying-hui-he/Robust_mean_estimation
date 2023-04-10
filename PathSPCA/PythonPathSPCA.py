#!/usr/bin/env python
from scipy import array,real,dot,column_stack,row_stack,append
import numpy
ra = numpy.random
la = numpy.linalg

def PathSPCA(A,k):
    M,N=A.shape
    # Loop through variables
    As=((A*A).sum(axis=0));vmax=As.max();vp=As.argmax();subset=[vp];
    vars=[];res=subset;rhos=[(A[:,vp]*A[:,vp]).sum()];
    Stemp=array([rhos])
    for i in range(1,k):
        lev,v=la.eig(Stemp)
        vars.append(real(lev).max())
        vp=real(lev).argmax()
        x=dot(A[:,subset],v[:,vp])
        x=x/la.norm(x)
        seto=range(0,N)
        for j in subset:
            seto.remove(j)
        vals=dot(x.T,A[:,seto]);vals=vals*vals
        rhos.append(vals.max())
        vpo=seto[vals.argmax()]
        Stemp=column_stack((Stemp,dot(A[:,subset].T,A[:,vpo])))
        vbuf=append(dot(A[:,vpo].T,A[:,subset]),array([(A[:,vpo]*A[:,vpo]).sum()]))
        Stemp=row_stack((Stemp,vbuf))
        subset.append(vpo)
    lev,v=la.eig(Stemp)
    vars.append(real(lev).max())
    return vars,res,rhos

# **** Run quick demo ****
# Simple data matrix with N=7 variables and M=3 samples
k=3 # target cardinality
A=array([[1,2,3,4,3,2,1],[4,2,1,4,3,2,1],[5,2,3,4,3,3,1]])

# Call function
vars,res,rhos=PathSPCA(A,k)
print res
print vars
print rhos
