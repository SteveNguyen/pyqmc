#!/usr/bin/python
# -*- coding: utf-8 -*-

########################################################################
#  File Name	: 'ProbUtils.py'
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

import sparray as sp
# import ndsparse as sp
import numpy as np


import scipy.interpolate as interpolate
from scipy import stats

from multiprocessing import Process, Queue, Manager
from multiprocessing.process import current_process
from multiprocessing import cpu_count
import time

import random


import copy


# _EPSILON=0.0001
_EPSILON=0.000
_INF=10e10


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap



def gaussian(x, mu, sig, nbsig=3.0):

    dist_x_mu=x-mu
    if abs(dist_x_mu)>sig*nbsig:
        return 0.0

    else:
        return (1.0/(sig*np.sqrt(2.0*np.pi)))*np.exp(-0.5*((dist_x_mu/sig)**2))#/50.0 #WTF?


def gaussian_dist(dist_x_mu, sig, nbsig=3.0):


    if abs(dist_x_mu)>sig*nbsig:
        return 0.0

    else:
        return (1.0/(sig*np.sqrt(2.0*np.pi)))*np.exp(-0.5*((dist_x_mu/sig)**2))#/50.0 #WTF?



def totuple(idx):


    if isinstance(idx, list):
        if len(idx) == 1:
            idx = (idx[0], )

    if not isinstance(idx, tuple):
        idx = (idx, )
    t = ()
    for i in idx:
        if isinstance(i, np.ndarray):
            t += (i[0], )
        else:
            t += (i, )


    return tuple(t)



def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    # print bin_edges.shape,bin_edges
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    # print bin_edges
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def RandomDraw(data,MIN,MAX,STEP,fill=0.0):

    data=np.array(data)
    data=data/data.sum()

    # print data
    # print np.cumsum(data)[1:],np.diff(np.arange(MIN,MAX,STEP)), np.arange(MIN,MAX,STEP)[1:]
    inv_cdf = interpolate.interp1d(np.cumsum(data)[1:]*np.diff(np.arange(MIN,MAX,STEP)), np.arange(MIN,MAX,STEP)[1:],bounds_error=False,fill_value=fill)

    # print min(np.cumsum(data)),max(np.cumsum(data)),random.uniform(min(np.cumsum(data)),max(np.cumsum(data)))
    # return inv_cdf(random.uniform(min(np.cumsum(data)),max(np.cumsum(data))))

    return inv_cdf(random.uniform(0.0,max(np.cumsum(data))))


def DiscreteRandomDraw(data,nb=1):

    data=np.array(data)
    if (np.isnan(data)).any():
        data = np.ones(len(data)) / (len(data))

    if data.sum() > 0:
        data=data/data.sum()
    xk = np.arange(len(data))

    custm = stats.rv_discrete(name='custm', values=(xk, data))
    return custm.rvs(size=nb)

class Distrib(sp.sparray):
    """ Tool class for handeling random variable. TODO copy"""

    def __init__(self, label='', shape=[], mins=[], maxs=[], circular=[],default=0.0, dtype=float):
        sp.sparray.__init__(self,shape,default,dtype)
        self.label=label
        self.mins=mins
        self.maxs=maxs
        self.circular=circular


        n=1
        for s in self.shape:
            n*=s

        self._num_elements=n #stupid, already done in sparray self.msize

    # def GetFlatFromIdx(self, index):
    #     """Turns index tuple into flat array *index*.

    #     This function performs the inverse of :func:`numpy.unravel_index`.
    #     """
    #     # Check bounds
    #     bounded_index=[]
    #     for i,ind in enumerate(index):
    #         if np.abs(ind)>=self.shape[i]:
    #             raise IndexError('index out of bounds')
    #         elif ind<0:
    #             bounded_index.append(self.shape[i]+ind)
    #         else:
    #             bounded_index.append(ind)

    #     return np.sum(np.asarray(bounded_index)*np.asarray(self.strides))


    def GetShape(self):
        return self.shape




    def GetFlatIdx(self,index):

        idx=0

        # if not isinstance(index, tuple) and not isinstance(index, list): #FIXME
        #     index = (index, )

        index = totuple(index)


        for dim in range(1,len(self.shape),1):
            idxout=index[dim-1]

            for dimout in range(dim,len(self.shape),1):

                # idxout *= self.shape[dimout]  #fucking python mutable stuff. Warning, do not try this at home, it will modifiy index in-place
                idxout=idxout*self.shape[dimout]


            idx+=idxout

        # print idx, index,

        idxflat = index[len(self.shape)-1]

        if isinstance(idxflat, list) or isinstance(idxflat, tuple):
            idxflat = idxflat[0]

        # print idx, idxflat
        # idx+=index[len(self.shape)-1]
        idx += idxflat

        # if not isinstance(idx,int):
        #     idx=idx[0]
        if isinstance(idx,list) or isinstance(idx,tuple):
            idx=idx[0]

        return idx


    def GetIdxFromFlat(self, index):


        if len(self.shape)==1:
            if not isinstance(index,tuple):
                return tuple([index])
            else:
                return index

        mod=1
        div=1
        idx=[]


        for dimout in xrange(len(self.shape)-1,-1,-1):

            mod=self.shape[dimout]
            idx.append(int(index/div)%mod)
            div*=self.shape[dimout]


        return tuple(idx[::-1]) #inverse order


    def Discretize(self,value):
        """ Return the discrete value (array) given a continuous value (array). TODO: dimension specific? """


        # if not isinstance(value,tuple):
        #     value=(value, )

        value = totuple(value)

        if len(value)==len(self.shape):

            for i in xrange(len(value)):

                v=value[i]
                if self.circular[i]==True:

                    while v>self.maxs[i]:
                        v-=(self.maxs[i] - self.mins[i])
                    while v<self.mins[i]:
                        v+=(self.maxs[i] - self.mins[i])#??


                d=0
                done=False

                ret=[]


                for it in np.linspace(self.mins[i],self.maxs[i],self.shape[i]+1):
                    if(v <= it):
                        if d==0:
                            ret.append(d)
                            done=True
                            break

                        elif done==False:
                            ret.append(d-1)
                            done=True
                            break

                    d+=1

                if done==False:
                    ret.append(d-2)

            return np.array(ret)



    def GetDist(self,x1,x2):
        """ Return the distance (continuous value) between two values. Handles circularity"""


        res=[]
        if not isinstance(x1,list):
            x1=[x1]
        if not isinstance(x2,list):
            x2=[x2]

        if len(x1)==len(self.shape) and len(x1)==len(x2):


            for i in xrange(len(x1)):
                dist=x1[i]-x2[i]
                if self.circular[i]==True:
                    if abs(dist) > abs(self.maxs[i]-self.mins[i])/2.0:
                        if dist<0.0:
                            dist+=(self.maxs[i]-self.mins[i])
                        elif dist>0.0:
                            dist-=(self.maxs[i]-self.mins[i])

                res.append(dist)
        return res



    def Continuize(self,value):
        """ Return the continuous value (array) given a discrete value (array). TODO: dimension specific? """


        #FIXME
        if isinstance(value,tuple):
            value=list(value)
        if not isinstance(value,list):
            value=[value]
        if len(value)==len(self.shape):

            v=value
            ret=[]
            for i in xrange(len(value)):


                if self.circular[i]==True:

                    v[i]%=self.shape[i]


                d=0
                done=False
                val=0.0
                # ret=[]


                for it in np.linspace(self.mins[i],self.maxs[i],self.shape[i]):

                    if v[i]==d:
                        ret.append(it)
                        done=True
                        break

                    d+=1
                    val=it

                if done==False:
                    ret.append(val)

            return np.array(ret)

    def GetNumElements(self):

        return self._num_elements

        # def GetProb(self, idx):
        #     """ Return the value of the proba from the idx tuble """
        #     return

        # def SetProb(self, idx, epsilon=_EPSILON):
        #     """ Set the value of the proba at the specified idx tuble. Default: do nothing if the proba is<epsilon (sparse) """
        #     return



class CondDistrib(Distrib):

    def __init__(self, llabel='', rlabel='', lshape=[], rshape=[], lmins=[], rmins=[], lmaxs=[],  rmaxs=[], lcircular=[], rcircular=[], default=0.0, dtype=float):

        self.LDistrib= Distrib(llabel, lshape, lmins, lmaxs, lcircular)
        self.RDistrib=Distrib(rlabel, rshape, rmins, rmaxs, rcircular)

        self.probs=sp.sparray(lshape+rshape,default,dtype)

        self.shape=self.probs.shape
        # self.strides=self.probs.strides

    def GetIterItems(self):
        return self.probs.GetIterItems()

    def NbNonZero(self):
        return self.probs.NbNonZero()

    def GetNumRElements(self):
        return self.RDistrib.GetNumElements()

    def GetNumLElements(self):
        return self.LDistrib.GetNumElements()

    def SetData(self,data):
        self.probs=data


    def GetNumElements(self):
        return self.probs.msize

    def GetProb(self,idx):
        return self.probs[idx]

    def SetProb(self,idx,value):

        if value>_EPSILON:
            # print idx #ICI
            self.probs[totuple(idx)]=value

    def GetProbLR(self,left,right):
        return self.probs[left+right]

    def SetProbLR(self,left,right,value):
        self.SetProb(left+right,value)

    def GetLShape(self):
        return self.LDistrib.GetShape()

    def GetRShape(self):
        return self.RDistrib.GetShape()


class Transition():
    """ Transition probability TODO"""

    def __init__(self,X,U, probs=None, StepBuild=None):

        self.X=X #These are Variables
        self.U=U
        # self.Xnext=Xnext
        self.probs=probs #by default should be P(X_next|X,U)

        self.StepBuild=StepBuild

        #set a uniform default value
        if self.probs != None:
            self.probs.__default = 1.0 / self.X.GetNumElements()


    def GetNbU(self):
        return self.U.GetNumElements()


    def GetNbX(self):
        return self.X.GetNumElements()

    def NbNonZero(self):
        return self.probs.NbNonZero()

    # def GetU(self, idx):
    #     return self.U[idx]

    # def GetX(self, idx):
    #     return self.X[idx]

    # def SetU(self, idx, val):
    #     self.U[idx]=val

    # def SetX(self, idx, val):
    #     self.X[idx]=val

    def GetProb(self, idx):
        # return self.probs[idx]
        return self.probs.GetProb(idx)

    def SetProb(self, idx, val):
        # self.probs[idx]=val
        self.probs.SetProb(idx,val)

    def GetIterItems(self):
        return self.probs.GetIterItems()


    def GetProbLR(self,left,right):
        return self.probs.GetProbLR(left,right)

    def SetProbLR(self,left,right, val):
        return self.probs.SetProbLR(left,right, val)


    def GetLShape(self):
        return self.probs.GetLShape()

    def GetRShape(self):
        return self.probs.GetRShape()


    def ProcessWorker(self, name, data, r):

        print "\tStarting Worker ",name

        for rx in r:

            # lpos,rpos,prob=self.StepBuild(rx)
            res=self.StepBuild(rx)

            # data[lpos+rpos]=prob
            # data=manager.dict(data.items()+res.items())
            data.update(res)
        return True


    @timing
    def ParallelBuild(self):


        if self.StepBuild == None:
            print "Please define StepBuild"
            return False

        print "building distrib"

        manager = Manager()
        shareddict=manager.dict()


        core_worker = cpu_count()
        print "Creating %d workers"%(core_worker)


        #cut the total range of element into subranges for each worker
        split_range=np.array_split(np.array(range(self.X.GetNumElements()*self.U.GetNumElements())),core_worker)

        #create workers
        workers = [Process(target=self.ProcessWorker, args=(i, shareddict, split_range[i])) for i in range(core_worker)]


        for each in workers:
            # print "Starting Worker"
            each.start()

        print "Waiting for process..."
        for each in workers:
            each.join()
            print "\tJoin() done"

        proba=sp.sparray(self.probs.shape,0.0,float,shareddict)
        # print shareddict
        print "Nb of non zero proba: ",len(shareddict)
        self.probs.SetData(proba)

        # print shareddict

    @timing
    def Build(self):

        data = dict()
        for x in range(self.X.GetNumElements()*self.U.GetNumElements()):
            res = self.StepBuild(x)
            data.update(res)
        proba=sp.sparray(self.probs.shape,0.0,float,data)
        print "Nb of non zero proba: ",len(data)
        self.probs.SetData(proba)


if __name__ == "__main__":

    A=Variable('', [10], [0.0], [1.0],[False])

    v=[0.8]

    r=A.Discretize(v)

    print r

    c=A.Continuize(r)

    print c

    A[0]=0.5
    A[5]=0.5

    print A
