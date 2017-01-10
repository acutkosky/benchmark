'''
Some learner classes to benchmark
'''

import numpy as np
import benchmark as bm

EPSILON = 0.000000000001

class OGD(bm.Learner):
    '''Online (sub)Gradient Descent
    hyperparmeter one_on_eta is the learning rate'''

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


def freeexp_diag_reg(w, scaling, k):
    '''regularizer used for diagonal freeexp'''
    abs_w = np.abs(w*scaling)
    return k * ((abs_w + 1)*np.log(abs_w + 1) - abs_w)

def freeexp_sphere_reg(w, k):
    ''''regularizer used for l2 freeexp'''
    norm_w = np.linalg.norm(w)
    return k * ((norm_w + 1)*np.log(norm_w + 1) - norm_w)

def update_learning_rate_sphere(accumulated_regret, old_L, one_over_eta_squared, \
    weights, gradient, gradients_sum, k, psi):
    '''Computes aggressive learning rate updates by measuring
    discrepencies between regret bounds.
    Will increase learning rates without compromising worst-case
    performance when possible.'''


    grad_norm = np.linalg.norm(gradient)
    L = np.maximum(old_L, grad_norm)
    if old_L == 0:
        old_L = L

    gradients_sum_norm = np.linalg.norm(gradients_sum)
    one_over_eta_plus_max = np.sqrt(np.maximum( \
            np.maximum(one_over_eta_squared - 2 * grad_norm \
                * np.minimum(old_L, grad_norm), \
            L * gradients_sum_norm), \
        2 * grad_norm**2 + EPSILON))

    one_over_eta_plus_min = np.sqrt(np.maximum(one_over_eta_squared + 2 * grad_norm \
            * np.minimum(old_L, grad_norm), \
        old_L * gradients_sum_norm))

    new_weights_plus_max = - (gradients_sum)/(gradients_sum_norm + EPSILON) \
        * (np.exp(gradients_sum_norm/(k * one_over_eta_plus_max)) - 1.0)
    new_weights_plus_min = - (gradients_sum)/(gradients_sum_norm + EPSILON) \
        * (np.exp(gradients_sum_norm/(k * one_over_eta_plus_min)) - 1.0)

    accumulated_regret_max = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_max) * psi(new_weights_plus_max) \
        + gradient * (weights - new_weights_plus_max)

    accumulated_regret_min = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_min) * psi(new_weights_plus_min) \
        + gradient * (weights - new_weights_plus_min)

    #Start with a Very Safe Learning Rate Update
    new_accumulated_regret = accumulated_regret_min
    new_one_over_eta_squared = np.maximum(one_over_eta_squared + 2*grad_norm**2, \
        L * gradients_sum_norm)

    if (accumulated_regret_max <= accumulated_regret_min).any():
        new_accumulated_regret = accumulated_regret_max
        new_one_over_eta_squared = one_over_eta_plus_max**2

    return new_accumulated_regret, new_one_over_eta_squared

def update_learning_rate_diag(accumulated_regret, old_L, one_over_eta_squared, \
    weights, gradient, gradients_sum, scaling, k, psi):
    '''Computes aggressive learning rate updates by measuring
    discrepencies between regret bounds.
    Will increase learning rates without compromising worst-case
    performance when possible.'''

    grad_norm = np.abs(gradient)
    L = np.maximum(old_L, grad_norm)
    old_L[old_L==0] = L[old_L==0]

    gradients_sum_norm = np.abs(gradients_sum)
    one_over_eta_plus_max = np.sqrt(np.maximum( \
            np.maximum(one_over_eta_squared - 2 * grad_norm \
                * np.minimum(old_L, grad_norm), \
            L * gradients_sum_norm), \
        2 * grad_norm**2 + EPSILON))

    one_over_eta_plus_min = np.sqrt(np.maximum(one_over_eta_squared + 2 * grad_norm \
            * np.minimum(old_L, grad_norm), \
        old_L * gradients_sum_norm))

    new_weights_plus_max = -np.sign(gradients_sum)/scaling \
        * (np.exp(gradients_sum_norm/(k * one_over_eta_plus_max)) - 1.0)
    new_weights_plus_min = -np.sign(gradients_sum)/scaling \
        * (np.exp(gradients_sum_norm/(k * one_over_eta_plus_min)) - 1.0)

    accumulated_regret_max = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_max) * psi(new_weights_plus_max) \
        + gradient * (weights - new_weights_plus_max)

    accumulated_regret_min = accumulated_regret \
        + (np.sqrt(one_over_eta_squared) - one_over_eta_plus_min) * psi(new_weights_plus_min) \
        + gradient * (weights - new_weights_plus_min)

    #Start with a Very Safe Learning Rate Update
    new_accumulated_regret = accumulated_regret_min
    new_one_over_eta_squared = np.maximum(one_over_eta_squared + 2*grad_norm**2, \
        L * gradients_sum_norm)

    increasable_indices = accumulated_regret_max <= accumulated_regret_min
    # Careful - we're overwriting accumulated_regret_min here!
    new_accumulated_regret[increasable_indices] = accumulated_regret_max[increasable_indices]
    new_one_over_eta_squared[increasable_indices] = one_over_eta_plus_max[increasable_indices]**2

    return new_accumulated_regret, new_one_over_eta_squared

class FreeExpSphere(bm.Learner):
    '''L2 FreeExp Learner'''
    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'k': 1.0}
        hyperparameters = {'k': hyperparameters['k']}

        super(FreeExpSphere, self).__init__('FreeExpSphere',hyperparameters)

        self.one_over_eta_squared = EPSILON
        self.one_over_eta_squared_without_increases = 0

        self.k = hyperparameters['k']
        self.L = 0
        self.gradients_sum = 0
        self.accumulated_regret = 0

        self.parameter = np.zeros(shape)

        self.psi = lambda w: freeexp_sphere_reg(w, self.k)

    def update(self, loss_info):
        '''update parameters'''
        super(FreeExpSphere, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.linalg.norm(gradient)
        gradients_sum_norm = np.linalg.norm(self.gradients_sum)

        self.accumulated_regret, new_one_over_eta_squared = \
            update_learning_rate_sphere(self.accumulated_regret, \
                self.L, self.one_over_eta_squared, \
                self.parameter, gradient, self.gradients_sum, self.k, self.psi)

        self.L = np.maximum(self.L, grad_norm)

        # compute a very safe learning rate update just for comparison
        self.one_over_eta_squared_without_increases = np.maximum(self.one_over_eta_squared \
            + 2*grad_norm**2, self.L * gradients_sum_norm)

        self.one_over_eta_squared = new_one_over_eta_squared

        self.extra_data = {'one_over_eta_squared': self.one_over_eta_squared, \
        'one_over_eta_squared_without_increases': self.one_over_eta_squared_without_increases}

        self.parameter = - self.gradients_sum/(gradients_sum_norm + EPSILON) \
            * (np.exp(gradients_sum_norm/(self.k * np.sqrt(self.one_over_eta_squared))) - 1.0)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeExpSphere, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f, 1/eta without increasing learning rates: %f' % \
            (np.sqrt(self.one_over_eta_squared), \
                np.sqrt(self.one_over_eta_squared_without_increases))
        return default_string + ' ' + increasing_learning_rates

    @staticmethod
    def hyperparameter_names():
        return ['k']

class FreeExpDiag(bm.Learner):
    '''diagonal FreeExp Learner'''
    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'k': 1.0}
        hyperparameters = {'k': hyperparameters['k']}
        super(FreeExpDiag, self).__init__('FreeExpDiag', hyperparameters)

        self.one_over_eta_squared = np.zeros(shape) + EPSILON
        self.one_over_eta_squared_without_increases = np.zeros(shape)

        self.k = hyperparameters['k']
        self.L = np.zeros(shape)
        self.gradients_sum = np.zeros(shape)
        self.accumulated_regret = np.zeros(shape)

        self.parameter = np.zeros(shape)

        self.scaling = np.ones(shape)

        self.psi = lambda weights: freeexp_diag_reg(weights, self.scaling, self.k)

    def update(self, loss_info):
        '''update parameters'''
        super(FreeExpDiag, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.abs(gradient)
        gradients_sum_norm = np.abs(self.gradients_sum)

        self.accumulated_regret, new_one_over_eta_squared = \
            update_learning_rate_diag(self.accumulated_regret, \
                self.L, self.one_over_eta_squared, \
                self.parameter, gradient, self.gradients_sum, self.scaling, self.k, self.psi)

        self.L = np.maximum(self.L, np.abs(gradient))

        # compute a very safe learning rate update just for comparison
        self.one_over_eta_squared_without_increases = np.maximum(self.one_over_eta_squared \
            + 2*grad_norm**2, self.L * np.abs(gradients_sum_norm))

        self.one_over_eta_squared = new_one_over_eta_squared

        self.extra_data = {'one_over_eta_squared': np.average(self.one_over_eta_squared), \
        'one_over_eta_squared_without_increases': np.average(self.one_over_eta_squared_without_increases)}

        self.parameter = -np.sign(self.gradients_sum)/self.scaling \
            * (np.exp(gradients_sum_norm/(self.k * np.sqrt(self.one_over_eta_squared))) - 1.0)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeExpDiag, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f, 1/eta without increasing learning rates: %f' % \
            (np.average(np.sqrt(self.one_over_eta_squared)), \
                np.average(np.sqrt(self.one_over_eta_squared_without_increases)))
        return default_string + ' ' + increasing_learning_rates

    @staticmethod
    def hyperparameter_names():
        return ['k']

class FreeExpScaledFeatures(FreeExpDiag):
    '''FreeExp that scales features to be theoretically robust to many measurements.'''

    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'k': 1.0}
        hyperparameters = {'k': hyperparameters['k']}
        super(FreeExpScaledFeatures, self).__init__('FreeExpScaledFeatures', hyperparameters)

        self.scaling = np.reshape(np.arange(1, 1+len(self.parameter.flatten())), shape)
        self.scaling = self.scaling * np.log(self.scaling + 1)
        self.psi = lambda weights: freeexp_diag_reg(weights, self.scaling, self.k)

    @staticmethod
    def hyperparameter_names():
        return ['k']

class PiSTOLSphere(bm.Learner):
    '''PiSTOL spherical learner'''

    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(PiSTOLSphere, self).__init__('PiSTOLSphere', hyperparameters)
        self.L = hyperparameters['L']
        self.parameter = np.zeros(shape)
        self.gradients_sum = np.zeros(shape)
        self.gradients_norm_sum = 0
        self.a = 2.25 * self.L
        self.b = 1

    @staticmethod
    def hyperparameter_names():
        return ['L']

    def update(self, loss_info):
        '''update parameters'''
        super(PiSTOLSphere, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.linalg.norm(gradient)
        self.gradients_norm_sum += grad_norm
        alpha = self.a * self.gradients_norm_sum + EPSILON
        self.parameter = - self.gradients_sum * self.b / alpha \
            * np.exp(np.linalg.norm(self.gradients_sum)**2/ (2 * alpha))

class PiSTOLDiag(bm.Learner):
    '''PiSTOL diagonal learner'''
    def __init__(self, shape, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(PiSTOLDiag, self).__init__('PiSTOLDiag', hyperparameters)
        self.L = hyperparameters['L']
        self.parameter = np.zeros(shape)
        self.gradients_sum = np.zeros(shape)
        self.gradients_norm_sum = np.zeros(shape)
        self.a = 2.25 * self.L
        self.b = 1

    @staticmethod
    def hyperparameter_names():
        return ['L']

    def update(self, loss_info):
        '''update parameters'''
        super(PiSTOLDiag, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.abs(gradient)
        self.gradients_norm_sum += grad_norm
        alpha = self.a * self.gradients_norm_sum + EPSILON
        self.parameter = - self.gradients_sum * self.b / alpha \
            * np.exp(np.abs(self.gradients_sum)**2/ (2 * alpha))

class PiSTOLScale(PiSTOLDiag):
    '''PiSTOL diagonal learner with features scaled by dimension'''
    def __init__(self, shape, hyperparameters=None):
        super(PiSTOLScale, self).__init__('PiSTOLDiag', shape, hyperparameters)
        self.b = 1.0/len(self.parameter.flatten())

class KTEstimatorSphere(bm.Learner):
    '''coin-betting based estimator using KT potential'''
    def __init__(self, shape, hyperparameters):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(KTEstimatorSphere, self).__init__('KTEstimatorSphere', hyperparameters)

        self.L = hyperparameters['L']
        self.eps = 1.0
        self.parameter = np.zeros(shape)
        self.loss_sum = 0
        self.gradients_sum = np.zeros(shape)
        self.t = 0


    @staticmethod
    def hyperparameter_names():
        return ['L']

    def update(self, loss_info):
        '''update parameters'''
        super(KTEstimatorSphere, self).update(loss_info)
        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        self.loss_sum += np.sum((gradient * self.parameter).flatten())
        self.t += 1
        self.parameter = - self.gradients_sum/self.t * (self.eps - self.loss_sum)

class KTEstimatorDiag(bm.Learner):
    '''diagonal coin-betting based estimator using KT potential'''
    def __init__(self, shape, hyperparameters):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(KTEstimatorDiag, self).__init__('KTEstimatorDiag', hyperparameters)

        self.L = hyperparameters['L']
        self.eps = 1.0
        self.parameter = np.zeros(shape)
        self.loss_sum = np.zeros(shape)
        self.gradients_sum = np.zeros(shape)
        self.t = 0


    @staticmethod
    def hyperparameter_names():
        return ['L']

    def update(self, loss_info):
        '''update parameters'''
        super(KTEstimatorDiag, self).update(loss_info)
        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        self.loss_sum += gradient * self.parameter
        self.t += 1
        self.parameter = - self.gradients_sum/self.t * (self.eps - self.loss_sum)

#set default use of FreeExp to be diag
FreeExp = FreeExpDiag

#set default use of PiSTOL to be diag
PiSTOL = PiSTOLDiag

#set default use of KTEstimator to be diag
KTEstimator = KTEstimatorDiag
