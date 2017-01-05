'''
Some learner classes to benchmark
'''

import numpy as np
import benchmark as bm

class OGD(bm.Learner):
    '''Online (sub-)Gradient Descent
    hyperparmeter eta is the learning rate'''

    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'eta': 1.0}
        hyperparameters = {'eta': hyperparameters['eta']}
        super(OGD, self).__init__('OGD', hyperparameters)
        self.eta = hyperparameters['eta']
        self.parameter = np.zeros(shape)

    def update(self, loss_info):
        '''updates parameters'''
        super(OGD, self).update(loss_info)

        gradient = loss_info['gradient']
        self.parameter -= self.eta * gradient
        return

class AdaGrad(bm.Learner):
    '''AdaGrad learner.
    Hyperparameter D is a scaling on the learning rate'''

    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'D': 1.0}
        hyperparameters = {'D': hyperparameters['D']}
        super(AdaGrad, self).__init__('AdaGrad', hyperparameters)
        self.D = hyperparameters['D']

        #add a negligible number to stave-off divide by zero errors
        self.sum_gradient_squared = np.zeros(shape) + 0.0000000000001

        self.parameter = np.zeros(shape)

    def update(self, loss_info):
        '''update parameters'''
        super(AdaGrad, self).update(loss_info)

        gradient = loss_info['gradient']

        self.sum_gradient_squared += gradient**2

        self.parameter -= self.D * gradient / np.sqrt(self.sum_gradient_squared)

