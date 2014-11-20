#!/usr/bin/python
# -*- coding: utf-8 -*-

########################################################################
#  File Name	: 'test.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: mercredi, février  5 2014
#  Revised	:
#  Version	:
#  Target MCU	:
#
#  This code is distributed under the GNU Public License
# 		which can be found at http://www.gnu.org/licenses/gpl.txt
#
#
#  Notes:	notes
########################################################################


from pyqmc.QMLib import *


from numpy import *
from scipy import *
import pylab as P
import matplotlib.pyplot as plt

import random as rd

import simple_lip as lip

import time



# _P_THRES=0.001

_P_THRES=0.00


XMIN=-pi
XMAX=pi

XCARD=11

VMIN=-2.5
VMAX=2.5

VCARD=11

UMIN=-0.5
UMAX=0.5

UCARD=5

SIGMA_POS=0.2
SIGMA_VEL=0.2



#Define the problem variables

Xpos=Distrib('Xpos', [XCARD], [XMIN], [XMAX], [True])
Xvel=Distrib('Xvel', [VCARD], [VMIN], [VMAX], [False])
U=Distrib('U', [UCARD], [UMIN], [UMAX], [False])
X=Distrib('XposXvel', [XCARD,VCARD], [XMIN,VMIN], [XMAX,VMAX], [True,False])

XposXvel_XposXvelU=CondDistrib('Xpos Xvel', 'Xpos Xvel U', [XCARD, VCARD], [XCARD, VCARD, UCARD]) #used to store the conditional FIXME

#used for the décomposition assuming independance
Xpos_XposXvelU=CondDistrib('Xpos', 'Xpos Xvel U',[XCARD], [XCARD, VCARD, UCARD])
Xvel_XposXvelU=CondDistrib('Xvel', 'Xpos Xvel U', [VCARD], [XCARD, VCARD, UCARD])



#define the custom fonction which return a dict containing the subrange of proba from a given right hand state
def ComputeStep(rx):

    xi,vi,ui=XposXvel_XposXvelU.RDistrib.GetIdxFromFlat(rx)
    # ui,vi,xi=XposXvel_XposXvelU.RDistrib.GetIdxFromFlat(rx)
    # print rx, xi, vi, ui
    x=Xpos.Continuize(xi)
    v=Xvel.Continuize(vi)
    u=U.Continuize(ui)


    Xnext=lip.simulate([x,v],[u])

    #put some Gaussian value around

    pxpos=0.0
    pxvel=0.0


    #dirty hack to keep track of the relevent non zero elements indexes
    nzpos_idx={}
    nzvel_idx={}


    res={}

    for li in xrange(Xpos_XposXvelU.GetNumLElements()): #iterate on Xpos only as we assume independance

        lidx=Xpos_XposXvelU.LDistrib.GetIdxFromFlat(li)

        # pxpos=gaussian(Xpos.Continuize(lidx),Xnext[0],SIGMA_POS)
        # pxvel=gaussian(Xvel.Continuize(lidx),Xnext[1],SIGMA_VEL)

        # print Xpos.GetDist(Xpos.Continuize(lidx),Xnext[0])[0], Xvel.GetDist(Xvel.Continuize(lidx),Xnext[1])[0]
        pxpos=gaussian_dist(Xpos.GetDist(Xpos.Continuize(lidx),Xnext[0])[0],SIGMA_POS)
        pxvel=gaussian_dist(Xvel.GetDist(Xvel.Continuize(lidx),Xnext[1])[0],SIGMA_VEL)





        # condidx=[xi,vi,ui,]+lidx
        # condidx=lidx+(xi,vi,ui)

        if pxpos > _P_THRES:
            nzpos_idx[lidx[0]]=pxpos  #in a general case it should be a tuple

        if pxvel > _P_THRES:
            nzvel_idx[lidx[0]]=pxvel


    for posidx in nzpos_idx:
        for velidx in nzvel_idx:

            # print posidx,velidx
            # XposXvel_XposXvelU.SetProbLR((posidx,velidx),(xi,vi,ui),nzpos_idx[posidx]*nzvel_idx[velidx])
            # return (posidx,velidx),(xi,vi,ui),nzpos_idx[posidx]*nzvel_idx[velidx]
            res[(posidx,velidx)+(xi,vi,ui)]=nzpos_idx[posidx]*nzvel_idx[velidx]

    return res


def ComputeStepAll(rx):

    xi,vi,ui=XposXvel_XposXvelU.RDistrib.GetIdxFromFlat(rx)
    # ui,vi,xi=XposXvel_XposXvelU.RDistrib.GetIdxFromFlat(rx)

    x=Xpos.Continuize(xi)
    v=Xvel.Continuize(vi)
    u=U.Continuize(ui)


    Xnext=lip.simulate([x,v],[u])

    #put some Gaussian value around

    pxpos=0.0
    pxvel=0.0


    #dirty hack to keep track of the relevent non zero elements indexes
    nzpos_idx={}
    nzvel_idx={}


    res={}

    for li in xrange(Xpos_XposXvelU.GetNumLElements()):
        for ri in xrange(Xvel_XposXvelU.GetNumLElements()):
            lidx=Xpos_XposXvelU.LDistrib.GetIdxFromFlat(li)
            ridx=Xvel_XposXvelU.LDistrib.GetIdxFromFlat(li)

        # pxpos=gaussian(Xpos.Continuize(lidx),Xnext[0],SIGMA_POS)
        # pxvel=gaussian(Xvel.Continuize(lidx),Xnext[1],SIGMA_VEL)

        # print Xpos.GetDist(Xpos.Continuize(lidx),Xnext[0])[0], Xvel.GetDist(Xvel.Continuize(lidx),Xnext[1])[0]
            pxpos=gaussian_dist(Xpos.GetDist(Xpos.Continuize(lidx),Xnext[0])[0],SIGMA_POS)
            pxvel=gaussian_dist(Xvel.GetDist(Xvel.Continuize(ridx),Xnext[1])[0],SIGMA_VEL)


            res[(posidx,velidx)+(xi,vi,ui)]=pxpos * pxvel

    return res




#We MUST put a main for multiprocessing

if __name__ == '__main__':



    #Define the transition probability

    # Tr=Transition(X,U,XposXvel_XposXvelU)
    Tr=Transition(X,U,XposXvel_XposXvelU,ComputeStep)
    #build the distribution
    Tr.ParallelBuild()
    # Tr.Build()


    # stop

    print "Building graph"
    quasi=QM(Tr)
    quasi.Init()

    quasi.Build()

    obj=tuple([int(XCARD/2),int(VCARD/2)])
    objidx=quasi.Vertices[obj]

    print "Computing quasi-distance"
    dist= quasi.Compute(objidx)

    # quasi.Save()



    dist=dist.reshape(XCARD,VCARD)
    print dist,dist.shape


    # fdist = open('dist_11x11x5.dat', 'w+')
    # save(fdist, dist)
    # fdist.close()


    # quasi.Policy((0,0))
    policy=quasi.ComputePolicyParallel()
    # policy=quasi.ComputePolicy()

    policy=policy.reshape(XCARD,VCARD)

    # fpol = open('pol_11x11x5.dat', 'w+')
    # save(fpol, policy)
    # fpol.close()


    # for i in range(100):
    #     print quasi.GetDrawnPolicy((5,5))

    print policy, policy.shape

    # quasi.Save()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    im=plt.imshow(dist.transpose(),origin='lower')
    im.set_interpolation('nearest')
    cb=plt.colorbar()
    ax.set_xlabel(r'Position')
    ax.set_ylabel(r'Velocity')

    plt.show()





    fig = plt.figure()
    ax = fig.add_subplot(111)
    im=plt.imshow(policy.transpose(),origin='lower')
    im.set_interpolation('nearest')
    cb=plt.colorbar()
    ax.set_xlabel(r'Position')
    ax.set_ylabel(r'Velocity')

    plt.show()
