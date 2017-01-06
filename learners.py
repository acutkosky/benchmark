'''
Some learner classes to benchmark
'''

import numpy as np
import benchmark as bm

EPSILON = 0.000000000001

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


def freeexp_diag_reg(w, scaling):
    '''regularizer used for diagonal freeexp'''
    abs_w = np.abs(w*scaling)
    return np.sqrt(5)*((abs_w + 1)*np.log(abs_w + 1) - abs_w)

def freeexp_sphere_reg(w):
    ''''regularizer used for l2 freeexp'''
    norm_w = np.linalg.norm(w)
    return np.sqrt(5)*((norm_w + 1)*np.log(norm_w + 1) - norm_w)

def update_learning_rate_sphere(accumulated_regret, old_L, one_over_eta_squared, \
    weights, gradient, sum_gradients, psi):
    '''Computes aggressive learning rate updates by measuring
    discrepencies between regret bounds.
    Will increase learning rates without compromising worst-case
    performance when possible.'''


    grad_norm = np.linalg.norm(gradient)
    L = np.maximum(old_L, grad_norm)
    if old_L == 0:
        old_L = L

    sum_grad_norm = np.linalg.norm(sum_gradients)
    one_over_eta_plus_max = np.sqrt(np.maximum( \
            np.maximum(one_over_eta_squared - 2 * grad_norm \
                * np.minimum(old_L, grad_norm), \
            L * sum_grad_norm), \
        2 * grad_norm**2 + EPSILON))

    one_over_eta_plus_min = np.sqrt(np.maximum(one_over_eta_squared + 2 * grad_norm \
            * np.minimum(old_L, grad_norm), \
        old_L * sum_grad_norm))

    new_weights_plus_max = - (sum_gradients)/(sum_grad_norm + EPSILON) \
        * (np.exp(sum_grad_norm/(np.sqrt(5) * one_over_eta_plus_max)) - 1)
    new_weights_plus_min = - (sum_gradients)/(sum_grad_norm + EPSILON) \
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

    if accumulated_regret_max <= accumulated_regret_min:
        new_accumulated_regret = accumulated_regret_max
        new_one_over_eta_squared = one_over_eta_plus_max**2

    return new_accumulated_regret, new_one_over_eta_squared

def update_learning_rate_diag(accumulated_regret, old_L, one_over_eta_squared, \
    weights, gradient, sum_gradients, scaling, psi):
    '''Computes aggressive learning rate updates by measuring
    discrepencies between regret bounds.
    Will increase learning rates without compromising worst-case
    performance when possible.'''

    grad_norm = np.abs(gradient)
    L = np.maximum(old_L, grad_norm)
    old_L[old_L==0] = L[old_L==0]

    sum_grad_norm = np.abs(sum_gradients)
    one_over_eta_plus_max = np.sqrt(np.maximum( \
            np.maximum(one_over_eta_squared - 2 * grad_norm \
                * np.minimum(old_L, grad_norm), \
            L * sum_grad_norm), \
        2 * grad_norm**2 + EPSILON))

    one_over_eta_plus_min = np.sqrt(np.maximum(one_over_eta_squared + 2 * grad_norm \
            * np.minimum(old_L, grad_norm), \
        old_L * sum_grad_norm))

    new_weights_plus_max = -np.sign(sum_gradients)/scaling \
        * (np.exp(sum_grad_norm/(np.sqrt(5) * one_over_eta_plus_max)) - 1)
    new_weights_plus_min = -np.sign(sum_gradients)/scaling \
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

    increasable_indices = accumulated_regret_max <= accumulated_regret_min
    # Careful - we're overwriting accumulated_regret_min here!
    new_accumulated_regret[increasable_indices] = accumulated_regret_max[increasable_indices]
    new_one_over_eta_squared[increasable_indices] = one_over_eta_plus_max[increasable_indices]**2

    return new_accumulated_regret, new_one_over_eta_squared

class FreeExpSphere(bm.Learner):
    '''L2 FreeExp Learner'''
    def __init__(self, shape, hyperparameters=None):
        hyperparameters = None
        super(FreeExpSphere, self).__init__('FreeExpSphere')

        self.one_over_eta_squared = EPSILON
        self.one_over_eta_squared_without_increases = 0

        self.L = 0
        self.sum_gradients = 0
        self.accumulated_regret = 0

        self.parameter = np.zeros(shape)

        self.psi = freeexp_sphere_reg

    def update(self, loss_info):
        '''update parameters'''
        super(FreeExpSphere, self).update(loss_info)

        gradient = loss_info['gradient']
        self.sum_gradients += gradient
        grad_norm = np.linalg.norm(gradient)
        sum_grad_norm = np.linalg.norm(self.sum_gradients)

        self.accumulated_regret, new_one_over_eta_squared = \
            update_learning_rate_sphere(self.accumulated_regret, \
                self.L, self.one_over_eta_squared, \
                self.parameter, gradient, self.sum_gradients, self.psi)

        self.L = np.maximum(self.L, grad_norm)

        # compute a very safe learning rate update just for comparison
        self.one_over_eta_squared_without_increases = np.maximum(self.one_over_eta_squared \
            + 2*grad_norm**2, self.L * sum_grad_norm)

        self.one_over_eta_squared = new_one_over_eta_squared

        self.parameter = - self.sum_gradients/(sum_grad_norm + EPSILON) \
            * (np.exp(sum_grad_norm/np.sqrt(5 * self.one_over_eta_squared)) - 1)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeExpSphere, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f, 1/eta without increasing learning rates: %f' % \
            (np.sqrt(self.one_over_eta_squared), \
                np.sqrt(self.one_over_eta_squared_without_increases))
        return default_string + ' ' + increasing_learning_rates

class FreeExpDiag(bm.Learner):
    '''diagonal FreeExp Learner'''
    def __init__(self, shape, hyperparameters=None):
        hyperparameters = None
        super(FreeExpDiag, self).__init__('FreeExpDiag', hyperparameters)

        self.one_over_eta_squared = np.zeros(shape) + EPSILON
        self.one_over_eta_squared_without_increases = np.zeros(shape)

        self.L = np.zeros(shape)
        self.sum_gradients = np.zeros(shape)
        self.accumulated_regret = np.zeros(shape)

        self.parameter = np.zeros(shape)

        self.scaling = np.ones(shape)

        self.psi = lambda weights: freeexp_diag_reg(weights, self.scaling)

    def update(self, loss_info):
        '''update parameters'''
        super(FreeExpDiag, self).update(loss_info)

        gradient = loss_info['gradient']
        self.sum_gradients += gradient
        grad_norm = np.abs(gradient)
        sum_grad_norm = np.abs(self.sum_gradients)

        self.accumulated_regret, new_one_over_eta_squared = \
            update_learning_rate_diag(self.accumulated_regret, \
                self.L, self.one_over_eta_squared, \
                self.parameter, gradient, self.sum_gradients, self.scaling, self.psi)

        self.L = np.maximum(self.L, np.abs(gradient))

        # compute a very safe learning rate update just for comparison
        self.one_over_eta_squared_without_increases = np.maximum(self.one_over_eta_squared \
            + 2*grad_norm**2, self.L * np.abs(sum_grad_norm))

        self.one_over_eta_squared = new_one_over_eta_squared

        self.parameter = -np.sign(self.sum_gradients)/self.scaling \
            * (np.exp(sum_grad_norm/np.sqrt(5 * self.one_over_eta_squared)) - 1)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeExpDiag, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f, 1/eta without increasing learning rates: %f' % \
            (np.average(np.sqrt(self.one_over_eta_squared)), \
                np.average(np.sqrt(self.one_over_eta_squared_without_increases)))
        return default_string + ' ' + increasing_learning_rates

class FreeExpScaledFeatures(FreeExpDiag):
    '''FreeExp that scales features to be theoretically robust to many measurements.'''

    def __init__(self, shape, hyperparameters=None):
        hyperparameters = None
        super(FreeExpScaledFeatures, self).__init__('FreeExpScaledFeatures', hyperparameters)

        self.scaling = np.reshape(np.arange(1, 1+len(self.parameter.flatten())), shape)
        self.scaling = self.scaling * np.log(self.scaling + 1)
        self.psi = lambda weights: freeexp_diag_reg(weights, self.scaling)
