def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import time
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster import RegularSpatial, KMeans
from tqdm import tqdm
from model import *
from collections import Counter

class BaseProtocol():
    @staticmethod
    def factory(name):
        if name == 'naive':
            return NaiveProtocol()
        if name == 'visited_score':
            return VisitedScoreProtocol()
        if name == 'greedy':
            return EpsilonGreedy()
        if name == 'ucb1':
            return UCB1()
        raise Exception
    
    def set_option(self, args):
        self.n_worker = args.n_worker
        self.n_replica = args.n_replica
        self.n_step = args.n_step
        self.n_cycle = args.n_cycle
        self.n_replica = args.n_replica
        self.output = Path('output')/self.name
        self.cluster_method = args.cluster_method
        self.t = 0
    
    def run_cycle(self, s):
        s.run_mc(self.n_step)
        return s.traj
    
    def run(self):
        print(f"running {self.name} protocol")
        self.trajs = []
        self.count = []
        self.t += 1
        for cycle_i in range(self.n_cycle):
            print("cycle {} started:".format(cycle_i))
            with Parallel(n_jobs=self.n_worker) as parallel:
                trajs = parallel(delayed(self.run_cycle)(s) for s in self.replica)
                for rep in range(self.n_replica):
                    fn = f'{cycle_i}_{rep}.npy'
                    np.save(self.output/fn, trajs[rep])
                    self.trajs.append(self.output/fn)
            self.discretize_trajs()
            self.seed()
            print("# of states found: {}".format(self.cluster.n_clusters_))

    def initialize(self):
        self.replica = [simulation() for _ in range(self.n_replica)]
        self.trajs = []
        self.dtrajs = []
        self.cluster_centers_ = None
        self.state_labels_ = None
        if self.cluster_method == 'regular':
            self.cluster = RegularSpatial(d_min=1)
        else:
            raise Exception
        self.msm = MarkovStateModel(lag_time=10, n_timescales=10, verbose=False)
    
    def discretize_trajs(self):
        if len(self.dtrajs) == 0:
            # initial analysis
            traj = [np.load(fn)[np.newaxis, :].T for fn in self.trajs]
            self.dtrajs = self.cluster.fit_transform(traj)
        else:
            # expand state definition
            begin = len(self.dtrajs)
            trajs = [np.load(fn)[np.newaxis, :].T for fn in self.trajs[begin:]]
            new_trajs = []
            has_new_trajs = False
            for traj in trajs:
                # select traj that does not belong to current cluster centers
                mask = np.all(np.sqrt(np.sum((traj[:, np.newaxis, :] - self.cluster.cluster_centers_)**2, axis=2)) > self.cluster.d_min, axis=1)
                if np.sum(mask) == 0:
                    continue
                new_trajs.append(traj[mask])
                has_new_trajs = True

            if has_new_trajs:
                cluster = RegularSpatial(d_min=self.cluster.d_min)
                cluster.fit(new_trajs)
                self.cluster.cluster_centers_ = np.concatenate((self.cluster.cluster_centers_, cluster.cluster_centers_))
                self.cluster.n_clusters_ += cluster.n_clusters_

            for traj in trajs:
                self.dtrajs.append(self.cluster.predict([traj])[0])
    
    def update_estimates(self):
        try:
            msm = MarkovStateModel(lag_time=10, n_timescales=10, verbose=False)
            msm.fit(self.dtrajs)
        except:
            msm = MarkovStateModel(lag_time=10, n_timescales=10, verbose=False, reversible_type='transpose')
            msm.fit(self.dtrajs)
        
        self.estimates = np.ones(self.cluster.n_clusters_)
        reward_states = np.intersect1d(self.reward_states, msm.state_labels_)
        if len(reward_states) > 0:
            for s_, s in enumerate(msm.state_labels_):
                if s in reward_states:
                    continue
                self.estimates[s] = np.sum(msm.transmat_[s_,:][reward_states])
        self.estimates /= np.sum(self.estimates)
    
    @property
    def reward_states(self):
        states = []
        for ix, c in enumerate(self.cluster.cluster_centers_):
            if c[0] > 80 and c[0] < 90:
                states.append(ix)
        return states
    
    def seed(self):
        pass


class NaiveProtocol(BaseProtocol):
    name = 'naive'

    def seed(self):
        for i, s in enumerate(self.replica):
            choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica)
            pos = self.cluster.cluster_centers_[choices[i]][0]
            s.set_position(pos)


class VisitedScoreProtocol(BaseProtocol):
    name = 'visited_score'

    def seed(self):
        counter = Counter(np.concatenate(self.dtrajs))
        score = [1./counter[i] for i in range(self.cluster.n_clusters_)]
        score /= np.sum(score)
        for i, s in enumerate(self.replica):
            choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica, p=score)
            pos = self.cluster.cluster_centers_[choices[i]][0]
            s.set_position(pos)


class EpsilonGreedy(BaseProtocol):
    name = 'epsilon'
    eps = 0.1

    def seed(self):
        self.update_estimates()
        for i, s in enumerate(self.replica):
            if np.random.random() < self.eps:
                choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica)
                pos = self.cluster.cluster_centers_[choices[i]][0]
                s.set_position(pos)
            else:
                #ix = np.argmax(self.estimates)
                choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica, p=score)
                pos = self.cluster.cluster_centers_[choices[i]][0]
                s.set_position(pos)


class UCB1(BaseProtocol):
    name = 'ucb1'

    def seed(self):
        self.update_estimates()
        counter = Counter(np.concatenate(self.dtrajs))
        score = [self.estimates[i] + np.sqrt(2 * np.log(self.t) / (1 + counter[i])) for i in range(self.cluster.n_clusters_)]
        score /= np.sum(score)
        for i, s in enumerate(self.replica):
            choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica, p=score)
            pos = self.cluster.cluster_centers_[choices[i]][0]
            print(f"new position: {pos}")
            s.set_position(pos)


class BayesianUCB(BaseProtocol):
    name = 'bayes_ucb'

    def seed(self):
        self.update_estimates()
        counter = Counter(np.concatenate(self.dtrajs))
        score = [self.estimates[i] + np.sqrt(2 * np.log(self.t) / (1 + counter[i])) for i in range(self.cluster.n_clusters_)]
        score /= np.sum(score)
        for i, s in enumerate(self.replica):
            choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica, p=score)
            pos = self.cluster.cluster_centers_[choices[i]][0]
            s.set_position(pos)
