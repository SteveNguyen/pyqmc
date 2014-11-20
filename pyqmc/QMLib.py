#!/usr/bin/python
# -*- coding: utf-8 -*-

########################################################################
#  File Name	: 'QMLib.py'
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

from ProbUtils import *

from graph_tool.all import *
# import multiprocessing as mp


from multiprocessing import Process, Queue, Manager
from multiprocessing.process import current_process
from multiprocessing import cpu_count

import copy
import time
import random
# import gc

import numpy as np

# try HOPE jit (unfortunately needs >= gcc 4.7)
#from hope import jit

# threshold on probability
_P_EPSILON = 0.0001
# _P_EPSILON=0.000
_INF = 10e10


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2 - time1) * 1000.0)
        return ret
    return wrap


class QM():

    """ Tool class for handeling the quasimetric """

    def __init__(self, tr, cost=lambda x, u: 1.0, LAMBDA=1.0, imle=None):

        self.Transition = tr
        self.G = Graph()

        self.Vertices = {}
        self.Vertices_pos = {}

        self.Edges = {}

        self.Vertex_pos = self.G.new_vertex_property("vector<int>")
        self.Edge_dist = self.G.new_edge_property("double")

        self.imle = imle
        self.GMM = None

        self.dist = []
        self.pred = []
        self.cost = cost

        # TODO make internal graph properties
        self.max_policy = []
        self.p_u = []
        self.beta = 0.01

        self.datacopy = []  # grr

        # for Lidstone experimental proba
        self.LAMBDA = LAMBDA
        self.Online_observations = {}

    def Init(self):

        # Should be a square matrix for transition probabilities from X to X
        # if self.Matrix != None:

        #     for xi in xrange(self.Matrix.shape[0]):

        #         v=self.G.add_vertex()
        #         self.Vertex_pos[v]=xi

        # Vertices[int(v)]=str([xi,vi])
        #         Vertices[xi]=int(v)

        # Better do that only when needed
        self.datacopy = copy.deepcopy(self.Transition.probs.probs)

        for xi in xrange(self.Transition.GetNbX()):

            v = self.G.add_vertex()
            self.Vertex_pos[v] = self.Transition.X.GetIdxFromFlat(xi)

            self.Vertices_pos[int(v)] = self.Transition.X.GetIdxFromFlat(xi)
            # transform into key
            self.Vertices[(self.Transition.X.GetIdxFromFlat(xi))] = int(v)

            self.Online_observations[int(v)] = {'action': {}, 'dest': {}}

        print "%d vertices" % (self.G.num_vertices())

    def Build(self):

        if self.imle != None:
            self.BuildGraphParallelIMLE()
        else:
            self.BuildGraphParallel2()

    # @timing
    def QuasiDistInit(self, orig, dest):

        opos = self.Vertices_pos[orig]
        dpos = self.Vertices_pos[dest]

        oidx = self.G.vertex(orig)
        didx = self.G.vertex(dest)

        p = 0.0
        mind = _INF

        # GetProbLR=self.Transition.GetProbLR

        data = self.Transition.probs.probs

        GetIdxFromFlat = self.Transition.U.GetIdxFromFlat

        # TODO iterate through non zero proba. How?
        for ui in xrange(self.Transition.GetNbU()):

            uidx = GetIdxFromFlat(ui)

            # p=self.Transition.GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))
            # p=GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))

            p = data[tuple(dpos) + tuple(opos) + tuple(uidx)]

            if p > _P_EPSILON:

                tmp = (self.cost(opos, uidx) / p)

                if tmp < mind:
                    mind = tmp

        if mind < _INF:
            return mind
        else:
            return -1

    # @timing
    def QuasiDistInit2(self, orig, dest, data):

        opos = self.Vertices_pos[orig]
        dpos = self.Vertices_pos[dest]

        p = 0.0
        mind = _INF

        # GetProbLR=self.Transition.probs.GetProbLR

        # data=copy.deepcopy(self.Transition.probs.probs)

        GetIdxFromFlat = self.Transition.U.GetIdxFromFlat

        # TODO iterate through non zero proba. How?
        for ui in xrange(self.Transition.GetNbU()):

            uidx = GetIdxFromFlat(ui)

            # p=self.Transition.GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))
            # p=GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))

            p = data[tuple(dpos) + tuple(opos) + tuple(uidx)]

            # idx=tuple(dpos)+tuple(opos)+tuple(uidx)
            # if idx in data.keys():
            # p=data[idx]

            if p > _P_EPSILON:

                tmp = (self.cost(opos, uidx) / p)

                if tmp < mind:
                    mind = tmp

        if mind < _INF:
            return mind
        else:
            return -1

    # @timing
    def QuasiDistInitIMLE(self, orig, dest):
        # deprecated

        opos = self.Vertices_pos[orig]
        dpos = self.Vertices_pos[dest]

        ostate = self.Transition.X.Continuize(opos)
        dstate = self.Transition.X.Continuize(dpos)

        p = 0.0
        mind = _INF

        # GetProbLR=self.Transition.probs.GetProbLR

        # data=copy.deepcopy(self.Transition.probs.probs)

        GetIdxFromFlat = self.Transition.U.GetIdxFromFlat

        lshape = self.Transition.GetLShape()
        rshape = self.Transition.GetRShape()

        # TODO iterate through non zero proba. How?
        for ui in xrange(self.Transition.GetNbU()):

            uidx = GetIdxFromFlat(ui)

            action = self.Transition.U.Continuize(uidx)

            gmmslice = np.concatenate((ostate, action))

            # print "TEST
            # ",action,gmmslice,gmmslice.shape,gmmslice.reshape(-1,1),np.array(dstate),range(0,len(rshape)),range(len(rshape),len(rshape)+len(lshape)),np.array(dstate)

            time1 = time.time()

            gmmpend = self.imle.to_gmm().inference(range(0, len(rshape)), range(
                len(rshape), len(rshape) + len(lshape)), gmmslice.reshape(-1, 1))

            time2 = time.time()
            # print 'inference took %0.3f ms' % ((time2-time1)*1000.0)

            # self.GMM.UpdateInference(range(0,len(rshape)), range(len(rshape),len(rshape)+len(lshape)))
            # gmmpend=self.GMM.GetInference(gmmslice.reshape(-1,1))

            # p=self.Transition.GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))
            # p=GetProbLR(tuple(dpos),tuple(opos)+tuple(uidx))

            # p=data[tuple(dpos)+tuple(opos)+tuple(uidx)]

            time1 = time.time()
            p = gmmpend.probability(np.array(dstate))
            time2 = time.time()
            # print 'proba took %0.3f ms' % ((time2-time1)*1000.0)

            # idx=tuple(dpos)+tuple(opos)+tuple(uidx)
            # if idx in data.keys():
            # p=data[idx]

            if p > _P_EPSILON:

                tmp = (self.cost(opos, uidx) / p)

                if tmp < mind:
                    mind = tmp

        if mind < _INF:
            return mind
        else:
            return -1

    @timing
    def BuildGraph(self):

        # much faster than BuildGraphP...

        it = 0
        prepour = 0.0
        pour = 0.0

        for orig in self.G.vertices():

            # just to informe the user...
            pour = int(float(it) / float(self.G.num_vertices()) * 100.0)
            if pour != prepour:
                prepour = pour
                print pour, '%'
            it += 1

            # print orig,self.G.num_vertices()
            for dest in self.G.vertices():

                dist = 0.0

                if orig != dest:
                    dist = self.QuasiDistInit(int(orig), int(dest))

                # print dist
                if dist != -1:  # inf?

                    # e=self.G.add_edge(orig,dest)
                    # INVERSION: this is normal
                    e = self.G.add_edge(dest, orig)
                    self.Edge_dist[e] = dist

    # @timing
    def ProcessWorker(self, name, input_queue, result_queue):

        print "\tStarting Worker ", name

        done = False

        for origi in input_queue:

            if origi == None:
                print "\tWorker %d exit" % (name)
                result_queue.put(None)
                done = True
                return True
                # break

            else:
                # print "."
                for dest in self.G.vertices():

                    # orig=self.Vertices[str(origi)]
                    # key
                    orig = self.Vertices[origi]
                    desti = self.Vertices_pos[int(dest)]

                    dist = 0.0

                    if orig != dest:
                        dist = self.QuasiDistInit(int(orig), int(dest))

                    if dist != -1:
                        # result_queue.put((origi,desti,dist))
                        # result_queue[str((origi,desti))]=distg
                        # key
                        result_queue[(origi, desti)] = dist

        print "\tWorker %d exit" % (name)
        result_queue.put(None)
        done = True
        return True

    def ProcessWorker2(self, name, input_queue, result_queue, data):

        print "\tStarting Worker ", name

        done = False

        for origi in input_queue:

            # print "."
            for dest in self.G.vertices():

                # orig=self.Vertices[str(origi)]
                # key
                orig = self.Vertices[origi]

                desti = self.Vertices_pos[int(dest)]

                dist = 0.0

                if orig != dest:
                    dist = self.QuasiDistInit2(int(orig), int(dest), data)

                if dist != -1:
                    result_queue.put((origi, desti, dist))
                    # result_queue[str((origi,desti))]=dist
                    # res[str((origi,desti))]=dist

        # result_queue.update(res)
        print "\tWorker %d exit" % (name)
        result_queue.put(None)
        done = True
        # collected=gc.collect()
        # print collected

        return True

    def ProcessWorkerIMLE(self, name, input_queue, result_queue):

        print "\tStarting Worker ", name

        done = False

        for origi in input_queue:

            # print "."
            for dest in self.G.vertices():

                # orig=self.Vertices[str(origi)]
                # key
                orig = self.Vertices[origi]

                desti = self.Vertices_pos[int(dest)]

                dist = 0.0

                if orig != dest:
                    dist = self.QuasiDistInitIMLE(int(orig), int(dest))

                if dist != -1:
                    result_queue.put((origi, desti, dist))
                    # result_queue[str((origi,desti))]=dist
                    # res[str((origi,desti))]=dist

        # result_queue.update(res)
        print "\tWorker %d exit" % (name)
        result_queue.put(None)
        done = True
        # collected=gc.collect()
        # print collected

        return True

    @timing
    def BuildGraphParallel(self):
        # deprecated

        it = 0
        prepour = 0.0
        pour = 0.0

        core_worker = cpu_count()
        print "Creating %d workers" % (core_worker)

        # let's put orig in the input queue and get the results in the output
        # queue

        input_queue = Queue()
        result_queue = Queue()

        manager = Manager()
        shareddict = manager.dict()

        in_queue_list = []
        # res_queue_list=[] #a list of output queue

        for i in range(core_worker):
            # input_queue.put(None)
            in_queue_list.append([])
            # in_queue_list.append(Queue())
            # res_queue_list.append(Queue())

        # cut the total range of element into subranges for each worker
        split_range = np.array_split(
            np.array(range(self.G.num_vertices())), core_worker)

        # we put in each input queue the list of vertices to be computed
        for i in range(core_worker):
            r = split_range[i]
            for v in r:
                in_queue_list[i].append((self.Vertices_pos[v]))
        workers = [Process(target=self.ProcessWorker, args=(
            i, in_queue_list[i], shareddict)) for i in range(core_worker)]

        # so launch workers in parallel to compute the initial quasi distance and return it through a queue
        # Then add these distance to the graph in a single thread (graph access
        # is not thread safe)
        for each in workers:
            # print "Starting Worker"
            each.start()

        print "Adding edges..."

        # Unfortunately we are limited to use a single process to actually write in the graph
        # Doesn't change if using multiple input and output queues.
        # TODO let's try with a kind of shared memory and a lock mechanism

        done = False
        nbdone = 0

        while not done:

            r = result_queue.get()

            if r == None:
                print "\tWorker finished?"
                nbdone += 1
                # print "rien??"

                if nbdone == core_worker:
                    done = True
                    break

            else:
                origi = r[0]
                desti = r[1]
                dist = r[2]

                # key
                orig = self.Vertices[origi]
                dest = self.Vertices[desti]

                if dist != -1:  # inf?

                    # e=self.G.add_edge(orig,dest)
                    # INVERSION (this is totally normal)
                    e = self.G.add_edge(dest, orig)
                    self.Edge_dist[e] = dist

        print "Adding edges done"
        input_queue.empty()
        result_queue.empty()

        print "\tJoin() ?"
        for each in workers:
            each.join(timeout=5)
            print "\tJoin() done"

    @timing
    def BuildGraphParallel2(self):

        # Parellel computing, dispatcher like

        core_worker = cpu_count()
        print "Creating %d workers" % (core_worker)

        # let's put orig in the input queue and get the results in the output
        # queue

        # input_queue = Queue()
        result_queue = Queue()

        # so slow...
        # manager = Manager()
        # shareddict=manager.dict(self.Transition.probs.probs.GetData())

        # F**king slow and memory consuming but still faster...
        # data=copy.deepcopy(self.Transition.probs.probs.GetData())
        # data=copy.deepcopy(self.Transition.probs.probs)
        # self.datacopy=data

        self.datacopy = copy.deepcopy(self.Transition.probs.probs)

        in_queue_list = []
        # res_queue_list=[] #a list of output queue

        for i in range(core_worker):
            # input_queue.put(None)
            in_queue_list.append([])
            # in_queue_list.append(Queue())
            # res_queue_list.append(Queue())

        # cut the total range of element into subranges for each worker
        split_range = np.array_split(
            np.array(range(self.G.num_vertices())), core_worker)

        # we put in each input queue the list of vertices to be computed
        for i in range(core_worker):
            r = split_range[i]
            for v in r:
                in_queue_list[i].append((self.Vertices_pos[v]))

        # poison pill
        # for i in range(core_worker):
        #     input_queue.put(None)

        workers = [Process(target=self.ProcessWorker2, args=(
            i, in_queue_list[i], result_queue, self.datacopy)) for i in range(core_worker)]

        # so launch workers in parallel to compute the initial quasi distance and return it through a queue
        # Then add these distance to the graph in a single thread (graph access
        # is not thread safe)
        for each in workers:
            # print "Starting Worker"
            each.start()

        print "Adding edges..."

        # Unfortunately we are limited to use a single process to actually write in the graph
        # Doesn't change if using multiple input and output queues.
        # TODO let's try with a kind of shared memory and a lock mechanism

        done = False
        nbdone = 0

        while not done:

            r = result_queue.get()

            if r == None:
                print "\tWorker finished?"
                nbdone += 1
                # print "rien??"

                if nbdone == core_worker:
                    done = True
                    break

            else:
                origi = r[0]
                desti = r[1]
                dist = r[2]

                # key
                orig = self.Vertices[origi]
                dest = self.Vertices[desti]

                if dist != -1:  # inf?

                    # e=self.G.add_edge(orig,dest)
                    # INVERSION (this is totally normal)
                    e = self.G.add_edge(dest, orig)
                    self.Edge_dist[e] = dist
                    self.Edges[(dest, orig)] = e

        print "\tJoin() ?"
        for each in workers:
            each.join()
            print "\tJoin() done"

        print "Adding edges done, %d edges" % (self.G.num_edges())
        # input_queue.empty()
        result_queue.empty()

    @timing
    def BuildGraphParallelIMLE(self):

        core_worker = cpu_count()
        print "Creating %d workers" % (core_worker)

        # let's put orig in the input queue and get the results in the output
        # queue

        # input_queue = Queue()
        result_queue = Queue()

        # prepare the GMM
        lshape = self.Transition.GetLShape()
        rshape = self.Transition.GetRShape()

        # self.GMM=self.imle.to_gmm()
        # self.GMM.UpdateInference(range(0,len(rshape)), range(len(rshape),len(rshape)+len(lshape)))

        # so slow...
        # manager = Manager()
        # shareddict=manager.dict(self.Transition.probs.probs.GetData())

        # F**king slow and memory consuming but still faster...
        # data=copy.deepcopy(self.Transition.probs.probs.GetData())
        # data=copy.deepcopy(self.Transition.probs.probs)
        # self.datacopy=data

        in_queue_list = []
        # res_queue_list=[] #a list of output queue

        for i in range(core_worker):
            # input_queue.put(None)
            in_queue_list.append([])
            # in_queue_list.append(Queue())
            # res_queue_list.append(Queue())

        # cut the total range of element into subranges for each worker
        split_range = np.array_split(
            np.array(range(self.G.num_vertices())), core_worker)

        # we put in each input queue the list of vertices to be computed
        for i in range(core_worker):
            r = split_range[i]
            for v in r:
                in_queue_list[i].append((self.Vertices_pos[v]))

        # poison pill
        # for i in range(core_worker):
        #     input_queue.put(None)

        workers = [Process(target=self.ProcessWorkerIMLE, args=(
            i, in_queue_list[i], result_queue)) for i in range(core_worker)]

        # so launch workers in parallel to compute the initial quasi distance and return it through a queue
        # Then add these distance to the graph in a single thread (graph access
        # is not thread safe)
        for each in workers:
            # print "Starting Worker"
            each.start()

        print "Adding edges..."

        # Unfortunately we are limited to use a single process to actually write in the graph
        # Doesn't change if using multiple input and output queues.
        # TODO let's try with a kind of shared memory and a lock mechanism

        done = False
        nbdone = 0

        while not done:

            r = result_queue.get()

            if r == None:
                print "\tWorker finished?"
                nbdone += 1
                # print "rien??"

                if nbdone == core_worker:
                    done = True
                    break

            else:
                origi = r[0]
                desti = r[1]
                dist = r[2]

                # key
                orig = self.Vertices[origi]
                dest = self.Vertices[desti]

                if dist != -1:  # inf?

                    # e=self.G.add_edge(orig,dest)
                    # INVERSION (this is totally normal)
                    e = self.G.add_edge(dest, orig)
                    self.Edge_dist[e] = dist
                    self.Edges[(dest, orig)] = e

        print "\tJoin() ?"
        for each in workers:
            each.join()
            print "\tJoin() done"

        print "Adding edges done, %d edges" % (self.G.num_edges())
        # input_queue.empty()
        result_queue.empty()

    @timing
    def BuildGraphP(self):
        # deprecated

        # Wow much slower than BuildGraph???

        it = 0
        prepour = 0.0
        pour = 0.0
        # TODO parallelize!!!

        # only non zero proba
        for idx, p in self.Transition.GetIterItems():

            pour = int(float(it) / float(self.Transition.NbNonZero()) * 100.0)
            if pour != prepour:
                prepour = pour
                print pour, '%'
            it += 1

            # idx should of the form of X(t+1),X(t),U(t)

            # dimensions of the proba (X part)
            lenx = len(self.Transition.X.shape)

            # cut the index for X(t+1) ans X(t)
            desti = list(idx[:lenx])
            origi = list(idx[lenx:2 * lenx])

            # ui=idx[2*lenx]
            # TODO let's find a way to considere only the relevant non zero U

            dist = 0.0

            # orig=self.Vertices[str(origi)]
            # dest=self.Vertices[str(desti)]

            # key
            orig = self.Vertices[origi]
            dest = self.Vertices[desti]

            if orig != dest:
                dist = self.QuasiDistInit(orig, dest)

                # print dist
            if dist != -1:  # inf?

                # e=self.G.add_edge(orig,dest)
                e = self.G.add_edge(dest, orig)  # INVERSION
                self.Edge_dist[e] = dist

    @timing
    def Compute(self, obj):  # TODO generic, obj independant

        self.dist, self.pred = dijkstra_search(
            self.G, self.G.vertex(obj), self.Edge_dist)

        return self.dist.a

    def PolicyWorker(self, name, statelist, data, maxpolicy, fullpolicy):

        print "\tStarting worker ", name

        for state in statelist:
            tsum = 0.0

            vertidx = self.Vertices[state]
            vert = self.G.vertex(vertidx)
            pz = vert.in_edges()

            dxy = self.dist[vert]

            # if dxy!=np.inf:

            pu = np.zeros(self.Transition.U.GetNumElements())
            for ui in xrange(self.Transition.U.GetNumElements()):
                pz = vert.in_edges()  # why the f**ck should I re-do that?
                msum = 0.0
                uidx = self.Transition.U.GetIdxFromFlat(ui)

                for z in pz:

                    zvert = z.source()

                    dest = self.Vertices_pos[int(zvert)]
                    # probably need to optimize this call
                    # print tuple(dest),tuple(state)+tuple(uidx)
                    # p=self.Transition.GetProbLR(tuple(dest),tuple(state)+tuple(uidx))
                    p = data[tuple(dest) + tuple(state) + tuple(uidx)]

                    if p > 0.0:
                        msum += self.dist[zvert] * p - dxy

                    else:
                        msum += -dxy

                msum += self.cost(state, uidx)

                tsum += msum

                puui = np.exp(-self.beta * msum)

                if puui != np.inf:
                    pu[ui] = puui
                else:
                    pu[ui] = _INF

            # n=np.exp(-self.beta*tsum)

            # if n!=0.0 and n!=np.inf:
            #     pu=pu/n
            # else:
            #     pu=pu*_INF

            norma = pu.sum()
            if norma != 0.0 and norma != np.inf:
                pu = pu / norma

            fullpolicy[int(vert)] = pu
            maxpol = None

            # test if uniform
            if not (pu == pu[0]).all():
                # maxpol=self.Transition.U.GetIdxFromFlat(np.argmax(pu))
                maxpol = np.argmax(pu)

            # else:
            #     print state, dxy

            maxpolicy[int(vert)] = maxpol

        print "\tWorker %d exit" % (name)
        return True

    def Policy(self, state):

        # in edges

        tsum = 0.0

        vertidx = self.Vertices[state]
        vert = self.G.vertex(vertidx)
        pz = vert.in_edges()

        dxy = self.dist[vert]

        if dxy != np.inf:

            pu = np.zeros(self.Transition.U.GetNumElements())
            for ui in xrange(self.Transition.U.GetNumElements()):
                pz = vert.in_edges()  # why the f**ck should I re-do that?
                sum = 0.0
                uidx = self.Transition.U.GetIdxFromFlat(ui)

                for z in pz:

                    zvert = z.source()

                    dest = self.Vertices_pos[int(zvert)]
                    # probably need to optimize this call
                    # print tuple(dest),tuple(state)+tuple(uidx)
                    p = self.Transition.GetProbLR(
                        tuple(dest), tuple(state) + tuple(uidx))

                    if p > 0.0:
                        sum += self.dist[zvert] * p - dxy

                    else:
                        sum += -dxy

                sum += self.cost(state, uidx)

                tsum += sum

                pu[ui] = np.exp(-self.beta * sum)

            # norma = np.exp(-self.beta*tsum)
            norma = pu.sum()
            if norma != 0.0 and norma != np.inf:
                pu = pu / norma

            maxpol = None

            # test if uniform
            if not (pu == pu[0]).all():
                # maxpol=self.Transition.U.GetIdxFromFlat(np.argmax(pu))
                maxpol = np.argmax(pu)

            # return pu,self.Transition.U.GetIdxFromFlat(np.argmax(pu))
            return pu, maxpol

        else:

            return np.ones(self.Transition.U.GetNumElements()) / self.Transition.U.GetNumElements(), None

    @timing
    def ComputePolicy(self):

        # print self.Transition.probs.probs.GetData()
        self.p_u = []
        self.max_policy = []
        for vert, pos in self.Vertices_pos.iteritems():

            p_u, minu = self.Policy(pos)
            self.p_u.append(p_u)
            self.max_policy.append(minu)

        self.max_policy = np.array(self.max_policy)
        self.p_u = np.array(self.p_u)
        return self.max_policy

    @timing
    def ComputePolicyParallel(self):

        print "Computing full policy"

        core_worker = cpu_count()
        print "Creating %d workers" % (core_worker)

        # TODO find a better way!
        # data=copy.deepcopy(self.Transition.probs.probs)
        # self.datacopy=data
        self.datacopy = copy.deepcopy(self.Transition.probs.probs)

        in_queue_list = []
        # res_queue_list=[] #a list of output queue

        manager = Manager()
        max_policy = manager.list(np.zeros(self.G.num_vertices()))
        full_policy = manager.list([[] for i in range(self.G.num_vertices())])

        for i in range(core_worker):

            in_queue_list.append([])

        # cut the total range of element into subranges for each worker
        split_range = np.array_split(
            np.array(range(self.G.num_vertices())), core_worker)

        # we put in each input queue the list of vertices to be computed
        for i in range(core_worker):
            r = split_range[i]
            for v in r:
                in_queue_list[i].append((self.Vertices_pos[v]))

        workers = [Process(target=self.PolicyWorker, args=(i, in_queue_list[
                           i], self.datacopy, max_policy, full_policy)) for i in range(core_worker)]

        for each in workers:
            # print "Starting Worker"
            each.start()

        print "\tJoin() ?"
        for each in workers:
            each.join()
            print "\tJoin() done"

        print "Policy done"

        self.max_policy = np.array(max_policy)
        self.p_u = np.array(full_policy)

        del max_policy
        del full_policy
        return self.max_policy

    def GetDrawnPolicy(self, state):
        # FIXME multiple dimension!
        pu = self.p_u[self.Transition.X.GetFlatIdx(state)]
        # u=[]

        if np.isnan(np.array(pu, dtype=float)).any():
            # problem
            # print pu
            return totuple(random.randrange(self.Transition.U.GetNumElements()))
        # for i in range(len(self.Transition.U.shape)):

        #     u.append(DiscreteRandomDraw(pu)[0])

        # return totuple(u)
        else:
            return totuple(DiscreteRandomDraw(pu)[0])

    def GetMaxPolicy(self, state):
        # FIXME multiple dimension!
        u = self.max_policy[self.Transition.X.GetFlatIdx(state)]
        # u=[]
        return totuple(u)

    def OnlineUpdate(self, orig, dest, action, cost):

        # print orig, dest, action
        # k=self.Transition.X.GetNumElements()
        k = self.Transition.GetNbX()

        # orig, dest and action are tuples

        origidx = self.Transition.X.GetFlatIdx(orig)
        destidx = self.Transition.X.GetFlatIdx(dest)

        actionidx = self.Transition.U.GetFlatIdx(action)

        # this state action has already been observed, add the cost to the cumulative cost
        # we will have for each XU: N*cost with N the number time XU was tested

        if actionidx in self.Online_observations[origidx]['action']:
            self.Online_observations[origidx][
                'action'][actionidx]['cost'] += cost
            self.Online_observations[origidx][
                'action'][actionidx]['total'] += 1.0

            # this transition has already been observed, increment the number of try
            # we will have for each XUX': M successful transition with
            # M=N*P(X'|XU)
            if destidx in self.Online_observations[origidx]['action'][actionidx]['transition']:
                self.Online_observations[origidx]['action'][
                    actionidx]['transition'][destidx]['success'] += 1.0

            else:  # just initialise it
                self.Online_observations[origidx]['action'][
                    actionidx]['transition'][destidx] = {}
                self.Online_observations[origidx]['action'][
                    actionidx]['transition'][destidx]['success'] = 1.0
                # self.Online_observations[origidx]['action'][actionidx]['transition'][destidx]['min']=None

        else:  # idem
            self.Online_observations[origidx]['action'][actionidx] = {}
            self.Online_observations[origidx][
                'action'][actionidx]['cost'] = cost
            self.Online_observations[origidx][
                'action'][actionidx]['total'] = 1.0

            self.Online_observations[origidx][
                'action'][actionidx]['transition'] = {}
            self.Online_observations[origidx]['action'][
                actionidx]['transition'][destidx] = {}
            self.Online_observations[origidx]['action'][
                actionidx]['transition'][destidx]['success'] = 1.0

        # let's check if we can update d0
        # as we keep track of N*cost and N*P(X'|XU) we can compute
        # cost/P(X'|XU)

        # d0=self.Online_observations[origidx]['action'][actionidx]['cost']/self.Online_observations[origidx]['action'][actionidx]['transition'][destidx]['success']

        # With Lidstone distrib
        meancost = self.Online_observations[origidx]['action'][actionidx][
            'cost'] / self.Online_observations[origidx]['action'][actionidx]['total']
        proba = (self.Online_observations[origidx]['action'][actionidx]['transition'][destidx][
                 'success'] + self.LAMBDA) / (self.Online_observations[origidx]['action'][actionidx]['total'] + self.LAMBDA * k)

        # let's keep track of the proba
        # TODO set also the other probas? The default value for a Transition is
        # set to 1/k
        self.Transition.SetProbLR(dest, orig + action, proba)

        d0 = meancost / proba

        # print
        # meancost,proba,d0,self.Online_observations[origidx]['action'][actionidx]['transition'][destidx]['success'],self.Online_observations[origidx]['action'][actionidx]['total']

        change = False
        if destidx in self.Online_observations[origidx]['dest']:
            # only if the experimental mean cost go down (min_u g/p)
            if d0 < self.Online_observations[origidx]['dest'][destidx]:
                self.Online_observations[origidx]['dest'][destidx] = d0
                change = True

            # else:
            # print 'dont change', d0,
            # self.Online_observations[origidx]['dest'][destidx], proba,
            # self.LAMBDA, self.LAMBDA * k
        else:
            self.Online_observations[origidx]['dest'][destidx] = d0
            change = True

        origvert = self.G.vertex(origidx)
        destvert = self.G.vertex(destidx)

        if change:
            if (destidx, origidx) in self.Edges:
                # print "Edge exists"
                # e=self.G.edge(destvert,origvert) #standard dest,orig
                # inversion

                # standard dest,orig inversion
                e = self.Edges[(destidx, origidx)]

                # in fact we allow this to change depending on observation
                # if self.Edge_dist[e] > float(d0):
                #     self.Edge_dist[e]=float(d0)
                self.Edge_dist[e] = float(d0)

            else:
                # print "Edge does not exist"
                e = self.G.add_edge(destvert, origvert)
                self.Edge_dist[e] = float(d0)
                # don't forget to add the edge here...
                self.Edges[(destidx, origidx)] = e

    def Save(self, name=None):

        if name != None:
            self.G.save(name)

        else:
            self.G.save("graph.xml.gz")
