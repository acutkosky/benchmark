'''
Benchmark online learning algorithms
'''

import time
import sys
import os

import numpy as np
from sklearn.datasets import load_svmlight_file

class Learner(object):
    '''Base class for building onlien learners'''
    def __init__(self, name, hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.parameter = None
        self.count = 0
        self.total_loss = 0
        self.total_gradient_norm = 0

    def __str__(self):
        return '%s;parameters=%s' % (self.name, str(self.hyperparameters))

    def __repr__(self):
        return 'Learner(Name: %s, Status: %s, hyperparameters: %s)' % \
            (self.name, self.get_status(), self.hyperparameters.__repr__())

    def update(self, loss_info):
        '''updates model parameters given loss_info dict.
        loss_info will contain:
        gradient: gradient at last prediction point
        loss: loss suffered at last prediction point

        it may contain other keys used by different algorithms (e.g. hessian info).
        '''
        self.total_loss += loss_info['loss']
        self.count += 1
        self.total_gradient_norm += np.linalg.norm(loss_info['gradient'])
        return

    def predict(self, predict_info):
        '''output a prediction based on input from predict_info, which is some context.
        In a pure online-learning setup, predict_info is always ignored.
        However, in other settings predict_info might be some potentially changing context.
        '''
        return self.parameter

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        if self.count == 0:
            return 'Not Started'
        else:
            return '%s: Updates: %d, Average loss: %f, Average Gradient: %f' % \
            (self.name, self.count, self.total_loss/(self.count), \
                self.total_gradient_norm/self.count)


class Lazy_dataset(object):
    '''Behaves like an object of class Dataset, but
    doesn't load data until accessed'''

    def __init__(self, name, loader):
        self.name = name
        self.loader = loader
        self.dataset = None

    def __getattr__(self, name):
        if self.dataset is None:
            self.dataset = self.loader()
            print 'Loaded Dataset '+self.name
        return getattr(self.dataset, name)

def permute_dataset(x_vals, y_vals):
    '''randomly permutes a dataset'''
    dataset_size = np.shape(x_vals)[0]
    permutation = np.random.permutation(dataset_size)
    return x_vals[permutation], y_vals[permutation]

class Dataset(object):
    '''Stores a dataset and provides access via loss functions'''
    def __init__(self, name, x_data, y_data, problem_type='regression', \
            permute=False, num_classes=None, process_labels=False):
        self.name = name
        self.problem_type = problem_type
        self.num_classes = num_classes
        assert self.problem_type == 'regression' or self.problem_type == 'classification'

        if self.problem_type == 'regression':
            self.loss_func = l2_loss
        else:
            self.loss_func = multiclass_hinge_loss

        self.x_data = x_data
        self.y_data = y_data
        self.dataset_size = np.shape(self.x_data)[0]

        if permute:
            self.x_data, self.y_data = permute_dataset(self.x_data, self.y_data)

        if self.problem_type == 'classification' and process_labels:
            relabeling = {}
            count = 0
            for class_index in np.unique(self.y_data):
                relabeling[class_index] = count
                count += 1
            for example_index in xrange(len(self.y_data)):
                self.y_data[example_index] = relabeling[self.y_data[example_index]]
        if self.problem_type == 'classification' and self.num_classes is None:
            self.num_classes = np.int(np.max(self.y_data)) + 1
        self.wshape = (self.num_classes, np.shape(self.x_data)[1])

        if self.problem_type == 'regression':
            self.wshape = (1, np.shape(self.x_data)[1])
            self.num_classes = None

        self.current_index = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Dataset(name=%s)' % (self.name)

    def get_example(self):
        '''gets the next example in the dataset, looping to beginning if needed'''
        x_val = self.x_data[self.current_index]
        y_val = self.y_data[self.current_index]
        self.current_index = (self.current_index+1) % (self.dataset_size)

        return x_val, y_val

    def examples(self):
        '''yield each example in the dataset in turn'''
        for index in xrange(self.dataset_size):
            yield self.x_data[index], self.y_data[index]

    def get_infos(self):
        '''yields predict_info, get_loss_info pairs'''
        for x_val, y_val in self.examples():
            yield x_val, lambda weights: self.loss_func(weights, x_val, y_val)


def load_libsvm_dataset(filename, name=None, problem_type='regression', \
        permute=False, num_classes=None):
    '''returns a lazy dataset for a given libsvm dataset file'''

    if name is None:
        name = os.path.basename(filename)

    return Lazy_dataset(name, lambda: Dataset(name, *load_svmlight_file(filename), \
        problem_type=problem_type, permute=permute, \
        num_classes=num_classes, process_labels=True))

def multiclass_hinge_loss(weights, features, label):
    '''computes multiclass hinge loss and its gradient

        weights: input prediction matrix
        features: input example to predict class for
        label: target class'''


    try:
        features = features.toarray()[0]
    except:
        pass
    prediction_scores = np.dot(weights, features)
    sorted_predictions = list(np.argsort(prediction_scores))
    if sorted_predictions[-1] != label:
        true_loss = 1.0
    else:
        true_loss = 0.0
    try:
        sorted_predictions.remove(np.int(label))
    except:
        print 'predictions: ', prediction_scores
        print 'label: ', label
        raise
    second_best_prediction = int(sorted_predictions[-1])
    label = int(label)

    gradient = np.zeros(np.shape(weights))
    hinge_loss = max(0.0, \
        1.0 + prediction_scores[second_best_prediction] - prediction_scores[label])
    if hinge_loss != 0:
        gradient[second_best_prediction] = features
        gradient[label] = -features
    return {'loss': true_loss, 'gradient': gradient, 'hinge_loss': hinge_loss}


def l2_loss(weights, features, label):
    '''computes squared error (w*x-y)^2 and its gradient with respect to w'''
    try:
        features = features.toarray()[0]
    except:
        pass

    prediction = features.dot(weights.T).flatten()[0]
    loss = (0.5*(prediction-label)**2).flatten()[0]
    gradient = (prediction-label)*features
    return {'loss': loss, 'gradient': gradient}



def run_learner(learner, dataset, status_interval=30):
    '''run a learner on a dataset, printing status
    every status_interval seconds'''
    print 'Uncached result - running experiment'
    start_time = time.time()
    losses = []
    total_loss = 0
    for predict_info, get_loss_info in dataset:
        loss_info = get_loss_info(learner.predict(predict_info))
        losses.append(loss_info['loss'])
        total_loss += loss_info['loss']

        learner.udpate(loss_info)

        if time.time() > start_time + status_interval:
            print "%s\r" % (learner.get_status())
            sys.stdout.flush()

    return {'learner': learner.name, \
            'losses': losses, \
            'total_loss': total_loss, \
            'hyperparameters': learner.hyperparameters}
