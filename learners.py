'''
Some learner classes to benchmark
'''

import numpy as np
import benchmark as bm

EPSILON = 0.000000000001

class OGD(bm.Learner):
    '''Online (sub)Gradient Descent
    hyperparmeter one_on_eta is the learning rate'''

    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'eta': 1.0}
        hyperparameters = {'eta': hyperparameters['eta']}
        super(OGD, self).__init__('OGD', hyperparameters)
        self.eta = hyperparameters['eta']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
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

    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'D': 1.0}
        hyperparameters = {'D': hyperparameters['D']}
        super(AdaGrad, self).__init__('AdaGrad', hyperparameters)
        self.D = hyperparameters['D']

    def dataset_initialize(self, dataset):

        shape = dataset.shape
        #add a negligible number to stave-off divide by zero errors
        self.sum_gradient_squared = np.zeros(shape) + EPSILON

        self.parameter = np.zeros(shape)

    def update(self, loss_info):
        '''update parameters'''
        super(AdaGrad, self).update(loss_info)

        gradient = loss_info['gradient']

        self.sum_gradient_squared += gradient**2

        self.parameter -= gradient / (self.D * np.sqrt(self.sum_gradient_squared))

    @staticmethod
    def hyperparameter_names():
        return ['D']

class FreeRexSphere(bm.Learner):
    '''L2 FreeExp Learner'''
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'k': 1.0}
        hyperparameters = {'k': hyperparameters['k']}

        super(FreeRexSphere, self).__init__('FreeRexSphere',hyperparameters)

        self.one_over_eta_squared = EPSILON

        self.k = hyperparameters['k']
        self.a = 1
        self.L = 0
        self.gradients_sum = 0

    def dataset_initialize(self, dataset):
        shape = dataset.shape
        self.parameter = np.zeros(shape)

    def update(self, loss_info):
        '''update parameters'''
        super(FreeRexSphere, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.linalg.norm(gradient)
        gradients_sum_norm = np.linalg.norm(self.gradients_sum)


        self.L = np.maximum(self.L, grad_norm)

        self.one_over_eta_squared = np.maximum(self.one_over_eta_squared + 2 * grad_norm**2,
                                                self.L * gradients_sum_norm)
        self.a = np.maximum(self.a, np.sqrt(self.one_over_eta_squared)/self.L)

        self.extra_data = {'one_over_eta_squared': self.one_over_eta_squared, \
        'a': self.a}

        self.parameter = - self.gradients_sum/(gradients_sum_norm * self.a + EPSILON) \
            * (np.exp(gradients_sum_norm/(self.k * np.sqrt(self.one_over_eta_squared))) - 1.0)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeRexSphere, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f' % (np.sqrt(self.one_over_eta_squared))
        return default_string + ' ' + increasing_learning_rates

    @staticmethod
    def hyperparameter_names():
        return ['k']

class FreeRexSphereMomentum(FreeRexSphere):
    def __init__(self, hyperparameters=None):
        super(FreeRexSphereMomentum, self).__init__(hyperparameters)
        self.name = 'FreeRexSphereMomentum'
        self.grad_norm_sum = 1.0

    def dataset_initialize(self, dataset):
        super(FreeRexSphereMomentum, self).dataset_initialize(dataset)
        shape = dataset.shape
        self.accumulated_parameters = np.zeros(shape)

    def update(self, loss_info):
        grad_norm = np.linalg.norm(loss_info['gradient'])
        self.grad_norm_sum += grad_norm
        self.accumulated_parameters += grad_norm * self.parameter
        super(FreeRexSphereMomentum, self).update(loss_info)

    def predict(self, prediction_info):
        return self.parameter + self.accumulated_parameters/self.grad_norm_sum

class FreeRexDiag(bm.Learner):
    '''diagonal FreeRex Learner'''
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'k': 1.0}
        hyperparameters = {'k': hyperparameters['k']}
        super(FreeRexDiag, self).__init__('FreeRexDiag', hyperparameters)
        self.k = hyperparameters['k']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
        self.one_over_eta_squared = np.zeros(shape) + EPSILON**2

        self.L = np.zeros(shape) + EPSILON
        self.gradients_sum = np.zeros(shape)

        self.parameter = np.zeros(shape)

        self.scaling = 1.0
        self.a = np.ones(shape)

        self.max_L2 = EPSILON

    def update(self, loss_info):
        '''update parameters'''
        super(FreeRexDiag, self).update(loss_info)

        gradient = loss_info['gradient']
        self.gradients_sum += gradient
        grad_norm = np.abs(gradient)
        gradients_sum_norm = np.abs(self.gradients_sum)

        self.L = np.maximum(self.L, grad_norm)

        self.max_L2 = np.maximum(self.max_L2, np.linalg.norm(gradient))

        self.one_over_eta_squared = np.maximum(self.one_over_eta_squared + 2 * grad_norm**2,
                                                self.L * gradients_sum_norm)
        self.a = np.maximum(self.a, np.sqrt(self.one_over_eta_squared)/self.L)

        self.scaling = np.maximum(self.scaling, np.sum(self.L)/self.max_L2)

        self.extra_data = {'one_over_eta_squared': np.average(self.one_over_eta_squared)}

        self.parameter = -np.sign(self.gradients_sum)/(self.scaling * self.a) \
            * (np.exp(gradients_sum_norm/(self.k * np.sqrt(self.one_over_eta_squared))) - 1.0)

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        default_string = super(FreeRexDiag, self).get_status()
        increasing_learning_rates = \
            '1/eta: %f' % (np.average(np.sqrt(self.one_over_eta_squared)))
        return default_string + ' ' + increasing_learning_rates

    @staticmethod
    def hyperparameter_names():
        return ['k']

class FreeRexDiagMomentum(FreeRexDiag):
    def __init__(self, hyperparameters=None):
        super(FreeRexDiagMomentum, self).__init__(hyperparameters)
        self.name = 'FreeRexDiagMomentum'

    def dataset_initialize(self, dataset):
        super(FreeRexDiagMomentum, self).dataset_initialize(dataset)
        shape = dataset.shape
        self.grad_norm_sum = np.ones(shape)
        self.accumulated_parameters = np.zeros(shape)

    def update(self, loss_info):
        grad_norm = np.abs(loss_info['gradient'])
        self.grad_norm_sum += grad_norm
        self.accumulated_parameters += grad_norm * self.parameter
        super(FreeRexDiagMomentum, self).update(loss_info)

    def predict(self, prediction_info):
        return self.parameter + self.accumulated_parameters/self.grad_norm_sum


class PiSTOLSphere(bm.Learner):
    '''PiSTOL spherical learner'''

    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(PiSTOLSphere, self).__init__('PiSTOLSphere', hyperparameters)
        self.L = hyperparameters['L']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
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
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(PiSTOLDiag, self).__init__('PiSTOLDiag', hyperparameters)
        self.L = hyperparameters['L']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
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

class PiSTOLScaledFeatures(PiSTOLDiag):
    '''PiSTOL diagonal learner with features scaled by dimension'''
    def __init__(self, hyperparameters=None):
        super(PiSTOLScaledFeatures, self).__init__(hyperparameters)
        self.name = 'PiSTOLScaledFeatures'

    def dataset_initialize(self, dataset):
        super(PiSTOLScaledFeatures, self).datset_initialize(dataset)
        self.b = 1.0/len(self.parameter.flatten())

    @staticmethod
    def hyperparameter_names():
        return ['L']


class KTEstimatorSphere(bm.Learner):
    '''coin-betting based estimator using KT potential'''
    def __init__(self, hyperparameters):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(KTEstimatorSphere, self).__init__('KTEstimatorSphere', hyperparameters)

        self.L = hyperparameters['L']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
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
        gradient = loss_info['gradient']/self.L
        self.gradients_sum += gradient
        self.loss_sum += np.sum((gradient * self.parameter).flatten())
        self.t += 1
        self.parameter = - self.gradients_sum/self.t * (self.eps - self.loss_sum)

class KTEstimatorDiag(bm.Learner):
    '''diagonal coin-betting based estimator using KT potential'''
    def __init__(self, hyperparameters):
        if hyperparameters is None:
            hyperparameters = {'L': 1.0}
        hyperparameters = {'L': hyperparameters['L']}
        super(KTEstimatorDiag, self).__init__('KTEstimatorDiag', hyperparameters)
        self.L = hyperparameters['L']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
        self.eps = 1.0
        self.parameter = np.zeros(shape)
        self.loss_sum = np.zeros(shape)
        self.gradients_sum = np.zeros(shape)
        self.scaling = 1.0
        self.t = 0


    @staticmethod
    def hyperparameter_names():
        return ['L']

    def update(self, loss_info):
        '''update parameters'''
        super(KTEstimatorDiag, self).update(loss_info)
        gradient = loss_info['gradient']/self.L
        self.gradients_sum += gradient
        self.loss_sum += gradient * self.parameter
        self.t += 1
        self.parameter = - self.gradients_sum/(self.t * self.scaling) * (self.eps - self.loss_sum)

class KTEstimatorScaledFeatures(KTEstimatorDiag):
    '''KTEstimator diagonal learner with features scaled by dimension'''
    def __init__(self, hyperparameters=None):
        super(KTEstimatorScaledFeatures, self).__init__(hyperparameters)
        self.name = 'KTEstimatorScaledFeatures'

    def dataset_initialize(self, dataset):
        super(KTEstimatorScaledFeatures, self).dataset_initialize(dataset)
        self.scaling = len(self.parameter.flatten())

    @staticmethod
    def hyperparameter_names():
        return ['L']

class MetaGradDiag(bm.Learner):
    '''diagonal metagrad learner (non-diagonal version takes quadratic time to update so I'm not going to make it.'''
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'DG': 1.0}
        hyperparameters = {'DG': hyperparameters['DG']}
        super(MetaGradDiag, self).__init__('MetaGradDiag', hyperparameters)

        self.eta_max =1.0/ hyperparameters['DG']

    def dataset_initialize(self, dataset):
        shape = dataset.shape
        self.T = dataset.dataset_size
        self.parameter = np.zeros(shape)
        self.num_etas = int(np.ceil(0.5*np.log(self.T)/np.log(2.0)))
        self.eta_values = [self.eta_max * np.power(2.0, -x) for x in xrange(self.num_etas)]
        self.eta_predictions = {eta:  np.zeros(shape) for eta in self.eta_values}

        priors = [1.0/((x+1.0)*(x+2)) for x in xrange(self.num_etas)]
        priors = priors/np.sum(priors)
        self.prior = {eta: prior for eta,prior in zip(self.eta_values, priors)}

        self.alpha = 1.0/len(self.parameter.flatten())

        self.M_sum = np.zeros(shape) + EPSILON

    def update(self, loss_info):
        super(MetaGradDiag, self).update(loss_info)

        gradient = loss_info['gradient']

        M = gradient**2
        self.M_sum += M
        ell_eta = lambda u, eta: eta*np.sum((u-self.parameter)*gradient) + eta**2 * np.sum((u-self.parameter)**2 * M)
        priors = {eta: self.prior[eta]*np.exp(-self.alpha * ell_eta(self.eta_predictions[eta], eta)) for eta in self.eta_values}
        normalization = np.sum(priors.values())
        self.prior = {eta: priors[eta]/normalization for eta in self.eta_values}

        for eta in self.eta_values:
            sigma = 1.0/self.eta_max * 2*eta**2 * self.M_sum
            old_prediction = self.eta_predictions[eta]
            new_prediction = old_prediction - 1.0/sigma * (eta*gradient + 2*eta**2 * M  * (old_prediction - self.parameter))
            self.eta_predictions[eta] = new_prediction

        tilted_normalization = np.sum([self.prior[eta]*eta for eta in self.prior])
        self.parameter = reduce(lambda acc,eta: acc + self.eta_predictions[eta]*eta*self.prior[eta], self.eta_values, 0)/tilted_normalization

    @staticmethod
    def hyperparameter_names():
        return ['DG']

#set default use of FreeExp to be diag
FreeRex = FreeRexDiag

#set default use of PiSTOL to be diag
PiSTOL = PiSTOLDiag

#set default use of KTEstimator to be diag
KTEstimator = KTEstimatorDiag
