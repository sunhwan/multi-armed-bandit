def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import time
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster import RegularSpatial
from tqdm import tqdm
from model import *
from collections import Counter

class Protocol():
    @staticmethod
    def factory(name):
        if name == 'naive':
            return NaiveProtocol()
        if name == 'visited_score':
            return VisitedScoreProtocol()
        if name == 'adaptive':
            return AdaptiveProtocol()
    
    def set_option(self, args):
        self.n_worker = args.n_worker
        self.n_replica = args.n_replica
        self.n_step = args.n_step
        self.n_cycle = args.n_cycle
        self.n_replica = args.n_replica
        self.output = Path('output')/self.name
    
    def run_cycle(self, s):
        s.run_mc(self.n_step)
        return s.traj
    
    def run(self):
        print(f"running {self.name} protocol")
        self.trajs = []
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
        self.cluster = RegularSpatial(d_min=1)
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
                cluster = RegularSpatial(d_min=1)
                cluster.fit(new_trajs)
                self.cluster.cluster_centers_ = np.concatenate((self.cluster.cluster_centers_, cluster.cluster_centers_))
                self.cluster.n_clusters_ += cluster.n_clusters_

            for traj in trajs:
                self.dtrajs.append(self.cluster.predict([traj])[0])
    
    def seed(self):
        pass


class NaiveProtocol(Protocol):
    name = 'naive'

    def seed(self):
        choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica)
        for i, s in enumerate(self.replica):
            pos = self.cluster.cluster_centers_[choices[i]][0]
            s.set_position(pos)


class VisitedScoreProtocol(Protocol):
    name = 'visited_score'

    def seed(self):
        counter = Counter(np.concatenate(self.dtrajs))
        score = [1/counter[i] for i in range(self.cluster.n_clusters_)]
        score /= np.sum(score)
        choices = np.random.choice(range(self.cluster.n_clusters_), self.n_replica, p=score)
        for i, s in enumerate(self.replica):
            pos = self.cluster.cluster_centers_[choices[i]][0]
            s.set_position(pos)
            

class AdaptiveProtocol(Protocol):
    name = 'adaptive'

    def seed(self):
        pass
    
