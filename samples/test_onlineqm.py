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


# import numpypy
from numpy import *
from scipy import *
import pylab as P
import matplotlib.pyplot as plt

import random as rd

import simple_lip as lip

import time
# import pp
# import sparray as spa

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

#the full conditionnal
XposXvel_XposXvelU=CondDistrib('Xpos Xvel', 'Xpos Xvel U', [XCARD, VCARD], [XCARD, VCARD, UCARD]) #used to store the conditional FIXME

#used for the décomposition assuming independance
Xpos_XposXvelU=CondDistrib('Xpos', 'Xpos Xvel U',[XCARD], [XCARD, VCARD, UCARD])
Xvel_XposXvelU=CondDistrib('Xvel', 'Xpos Xvel U', [VCARD], [XCARD, VCARD, UCARD])





def simulate(qm, nb=10000):
    print 'Simulation'

    for i in xrange(nb):

        if (i % 1000) == 0:
            print i

        xt=rd.uniform(XMIN,XMAX)
        dxt=rd.uniform(VMIN,VMAX)
        ut=rd.uniform(UMIN,UMAX)

        res=lip.simulate([xt,dxt],[ut])
        # xt1=res[0]+random.randn()*0.01 #+rd.gauss(0,0.01)
        # dxt1=res[1]+random.randn()*0.01#+rd.gauss(0,0.01)

        xt1=rd.gauss(res[0], 0.01)
        dxt1=rd.gauss(res[1], 0.01)



        xpos=Xpos.Discretize((xt))
        xvel=Xvel.Discretize((dxt))
        u=U.Discretize((ut))

        xposres=Xpos.Discretize((xt1))
        xvelres=Xvel.Discretize((dxt1))

        # print 'SIMU ',(xpos,xvel),(xposres,xvelres),(u)
        # print xt,dxt,ut,res[0],res[1],xt1,dxt1
        qm.OnlineUpdate((xpos,xvel),(xposres,xvelres),(u, ),1.0)


def simulate_all(qm,nb=100):
    print 'Simulation'

    for i in xrange(nb):
        print i



        for xi in xrange(XCARD):

            for vi in xrange(VCARD):
                for ui in xrange(UCARD):


                    xt = Xpos.Continuize(xi)
                    dxt = Xvel.Continuize(vi)
                    ut = U.Continuize(ui)
                    res=lip.simulate([xt,dxt],[ut])
                    # xt1=res[0]+random.randn()*0.01 #+rd.gauss(0,0.01)
                    # dxt1=res[1]+random.randn()*0.01#+rd.gauss(0,0.01)

                    # xt1=res[0]+rd.gauss(0,0.01)
                    # dxt1=res[1]+rd.gauss(0,0.01)

                    # xt1=res[0]
                    # dxt1=res[1]


                    xt1=rd.gauss(res[0], SIGMA_POS)
                    dxt1=rd.gauss(res[1], SIGMA_VEL)

                    # n1 = rd.gauss(0, SIGMA_POS)
                    # n2 = rd.gauss(0, SIGMA_VEL)

                    # xt1=res[0] + n1
                    # dxt1=res[1] + n2


                    # xt1=rd.gauss(res[0], 0.01)
                    # dxt1=rd.gauss(res[1], 0.01)

                    # xt1 = res[0]
                    # dxt1 = res[1]


                    # print xi, vi, xt, dxt,res, xt1, dxt1

                    # xpos=Xpos.Discretize((xt))
                    # xvel=Xvel.Discretize((dxt))
                    # u=U.Discretize((ut))

                    xpos = xi
                    xvel = vi
                    u = ui

                    xposres=Xpos.Discretize((xt1))
                    xvelres=Xvel.Discretize((dxt1))

                    # xposres=Xpos.Discretize((res[0] + n1))
                    # xvelres=Xvel.Discretize((res[1] + n2))


                    # print 'SIMU ',(xpos,xvel),(xposres,xvelres),(u)
                    # print xt,dxt,ut,res[0],res[1],xt1,dxt1
                    qm.OnlineUpdate((xpos,xvel),(xposres,xvelres),(u, ),1.0)






def cost(x,u):
    return 1.0

#We MUST put a main for multiprocessing

if __name__ == '__main__':


    #Define the transition probability
    # Tr=Transition(X,U,XposXvel_XposXvelU)
    #still usefull for onlineqm to configure dimensions and cardinality and also store the proba
    Tr=Transition(X,U,XposXvel_XposXvelU,None)

    #build the distribution: we don't need to do that for onlineqm as we experimentally learn the proba
    # Tr.ParallelBuild()


    print "Building graph"
    # quasi=OnlineQM(Tr,cost)
    quasi=QM(Tr,cost)
    quasi.Init()


    #here we simulate the system to learn
    # simulate(quasi,1000000)
    simulate(quasi,50000)
    # simulate_all(quasi)

    #we don't need to build, it was done online
    # quasi.Build()


    print "Adding edges done, %d edges"%(quasi.G.num_edges())


    obj=tuple([int(XCARD/2),int(VCARD/2)])
    objidx=quasi.Vertices[obj]

    print "Computing quasi-distance"
    dist= quasi.Compute(objidx)

    # quasi.Save()

    # print dist,dist.shape

    dist=dist.reshape(XCARD,VCARD)



    # quasi.Policy((0,0))
    policy=quasi.ComputePolicyParallel()
    # policy=quasi.ComputePolicy()

    policy = array(policy, dtype=float) #just in case we have some nan (= uniform distribs)
    policy=policy.reshape(XCARD,VCARD)

    print policy

    # stop
    # for i in range(100):
    #     print quasi.GetDrawnPolicy((5,5))



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
