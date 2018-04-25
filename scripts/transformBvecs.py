import numpy as np
bval=np.loadtxt('bval')
bvalsr=[]
for x in bval:
    bvalsr=np.append(bvalsr,[x, x ,x])
np.savetxt('bvalsr',bvalsr)

bvec=np.loadtxt('bvec')
bvec1=bvec[0]
bvec2=bvec[1]
bvec3=bvec[2]

bvec11=[]
bvec22=[]
bvec33=[]
for x in bvec1:
    bvec11=np.append(bvec11,[x, x ,x])
for x in bvec2:
    bvec22=np.append(bvec22,[x, x ,x])
for x in bvec3:
    bvec33=np.append(bvec33,[x, x ,x])

bvecsr=[]
bvecsr=np.append([bvec11,bvec22],[bvec33],axis=0)
np.savetxt('bvecsr',bvecsr)
