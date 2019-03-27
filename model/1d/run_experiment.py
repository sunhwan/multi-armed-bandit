import argparse
from pathlib import Path

from protocol import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='naive', choices=['naive', 'visited_score', 'greedy', 'ucb1'])
    parser.add_argument('--n_worker', default=10, type=int)
    parser.add_argument('--n_replica', default=10, type=int)
    parser.add_argument('--n_cycle', default=200, type=int)
    parser.add_argument('--n_step', default=1000, type=int)
    parser.add_argument('--cluster_method', default='regular', choices=['regular', 'kmeans'])
    parser.add_argument('--outputdir', default='output')
    args = parser.parse_args()

    basedir = Path(args.outputdir)
    prefix = f'{args.protocol}_{args.n_replica}_{args.n_cycle}_{args.n_step}'
    outputdir = basedir/prefix
    outputdir.mkdir(parents=True, exist_ok=True)

    protocol = BaseProtocol.factory(args.protocol)
    protocol.set_option(args)
    protocol.output = outputdir
    protocol.initialize()
    protocol.run()
   