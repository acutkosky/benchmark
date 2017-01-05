'''
Some learner classes to benchmark
'''

import numpy as np
import benchmark as bm

EPSILON = 0.0000000001

class OGD(bm.Learner):
    '''Online (sub)Gradient Descent
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

    @staticmethod
    def hyperparameter_names():
        return ['eta']

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
        self.sum_gradient_squared = np.zeros(shape) + EPSILON

        self.parameter = np.zeros(shape)

    def update(self, loss_info):
        '''update parameters'''
        super(AdaGrad, self).update(loss_info)

        gradient = loss_info['gradient']

        self.sum_gradient_squared += gradient**2

        self.parameter -= self.D * gradient / np.sqrt(self.sum_gradient_squared)

    @staticmethod
    def hyperparameter_names():
        return ['D']


def freeexp_reg(weights, scaling):
    abs_weights = np.abs(weights*scaling)
    return np.sqrt(5)*((abs_weights + 1)*np.log(abs_weights + 1) - abs_weights)

def update_learning_rate(accumulated_regret, old_L, one_over_eta_squared, \
    weights, gradient, sum_gradients, psi):
    '''Computes aggressive learning rate updates by measuring
    discrepencies between regret bounds.
    Will increase learning rates without compromising worst-case
    performance when possible.'''

    grad_norm = np.abs(gradient)
    L = np.maximum(old_L, grad_norm)
    sum_grad_norm = np.abs(sum - gradient)
    one_over_eta_plus_max = np.sqrt(np.maximum( \
            np.maximum(one_over_eta_squared - 2 * grad_norm \
                * np.minimum(old_L, grad_norm), \
            L * sum_grad_norm), \
        2 * grad_norm + EPSILON))

    one_over_eta_plus_min = np.sqrt(np.maximum(one_over_eta_squared + 2 * grad_norm \
            * np.minimum(old_L, grad_norm), \
        old_L * sum_grad_norm))

    new_weights_plus_max = -np.sign(sum_gradients) \
        * (np.exp(sum_grad_norm/(np.sqrt(5) * one_over_eta_plus_max)) - 1)
    new_weights_plus_min = -np.sign(sum_gradients) \
        * (np.exp(sum_grad_norm/(np.sqrt(5) * one_over_eta_plus_min)) - 1)

    accumulated_regret_max = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_max) * psi(new_weights_plus_max) \
        + gradient * (weights - new_weights_plus_max)

    accumulated_regret_min = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_min) * psi(new_weights_plus_min) \
        + gradient * (weights - new_weights_plus_min)

    #Start with a Very Safe Learning Rate Update
    new_accumulated_regret = accumulated_regret_min
    new_one_over_eta_squared = np.maximum(one_over_eta_squared + 2*grad_norm**2, \
        L * sum_grad_norm)

    increasable_indices = accumulated_regret_max <= 0
    # Careful - we're overwriting accumulated_regret_min here!
    new_accumulated_regret[increasable_indices] = accumulated_regret_max[increasable_indices]
    new_one_over_eta_squared[increasable_indices] = one_over_eta_plus_max[increasable_indices]**2

    return new_accumulated_regret, new_one_over_eta_squared


class FreeExp(bm.Learner):
    '''FreeExp Learner'''
    def __init__(self, shape):
        hyperparameters = None
        super(FreeExp, self).__init__('FreeExp', hyperparameters)

        self.one_over_eta_squared = np.zeros(shape) + EPSILON
        self.one_over_eta_squared_without_increases = np.zeros(shape)

        self.L = np.zeros(shape)
        self.sum_gradients = np.zeros(shape)
        self.accumulated_regret = np.zeros(shape)

        self.parameter = np.zeros(shape)

        scaling = np.reshape(np.arange(1, len(self.parameter.flatten())), shape)
        self.psi = lambda weights: freeexp_reg(weights, scaling)

    def update(self, loss_info):
        '''update parameters'''
        super(FreeExp, self).update(loss_info)

        gradient = loss_info['gradient']
        self.sum_gradients += gradient
        grad_norm = np.abs(gradient)
        sum_grad_norm = np.abs(self.sum_gradients)

        self.accumulated_regret, new_one_over_eta_squared = \
            update_learning_rate(self.accumulated_regret, \
                self.L, self.one_over_eta_squared, \
                self.parameter, gradient, self.sum_gradients, self.psi)

        self.L = np.maximum(self.L, np.abs(gradient))

        # compute a very safe learning rate update just for comparison
        self.one_over_eta_squared_without_increases = np.maximum(self.one_over_eta_squared \
            + 2*grad_norm**2, self.L * np.abs(sum_grad_norm))

        self.one_over_eta_squared = new_one_over_eta_squared

        self.parameter = -np.sign(self.sum_gradients) \
            * (np.exp(sum_grad_norm/np.sqrt(5 * self.one_over_eta_squared)) - 1)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeExp, self).get_status()
        increasing_learning_rates = \
            '1/eta^2: %f, 1/eta^2 without increasing learning rates: %f' % \
            (np.average(self.one_over_eta_squared), \
                np.average(self.one_over_eta_squared_without_increases))
        return default_string + ' ' + increasing_learning_rates
