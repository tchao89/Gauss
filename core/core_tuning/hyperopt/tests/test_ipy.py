'''
To use this test script, there should be a cluster of ipython parallel engines
instantiated already. Their working directory should be the current
directory: hyperopt/tests

To start the engines in hyperopt/hyperopt/tests/
  use: $ ipcluster start --n=2


'''
from __future__ import print_function
import sys
from nose import SkipTest
try:
    from IPython.parallel import Client
except ImportError:
    print("Skipping IPython Tests (IPython not found)", file=sys.stderr)
    raise SkipTest('IPython not present')

from core.core_tuning.hyperopt.ipy import IPythonTrials
from core.core_tuning import hyperopt


def test0():
    try:
        client = Client(debug=True)
    except IOError:
        raise SkipTest()

    client[:].use_dill()
    trials = IPythonTrials(client, 'log')

    def simple_objective(args):
        # -- why are these imports here !?
        # -- is it because they need to be imported on the client?
        #
        # Yes, the client namespace is empty, so some imports may be
        # needed here. Errors on the engines can be found by
        # using debug=True when instantiating the Client.
        return {'loss': args ** 2, 'status': hyperopt.STATUS_OK}

    space = core.hyperopt.hp.uniform('x', 0, 1)

    minval = trials.fmin(simple_objective, space=space,
                         algo=core.hyperopt.tpe.suggest, max_evals=25, verbose=True)
    print(minval)
    assert minval['x'] < .2


def test_fmin_fn():
    try:
        client = Client()
    except IOError:
        raise SkipTest()

    client[:].use_dill()

    trials = IPythonTrials(client, 'log')
    assert not trials._testing_fmin_was_called

    def simple_objective(args):
        return {'loss': args ** 2, 'status': hyperopt.STATUS_OK}

    space = core.hyperopt.hp.uniform('x', 0, 1)

    minval = hyperopt.fmin(simple_objective, space=space,
                           algo=core.hyperopt.tpe.suggest,
                           max_evals=25,
                           trials=trials)

    assert minval['x'] < .2
    assert trials._testing_fmin_was_called
