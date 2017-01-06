'''
Benchmark online learning algorithms
'''

import time
import sys
import os
import pickle
import re

import numpy as np
from sklearn.datasets import load_svmlight_file

import cachelog

class Learner(object):
    '''Base class for building online learners'''
    def __init__(self, name, hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.parameter = None
        self.count = 0
        self.total_loss = 0
        self.total_gradient_norm = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '%s;parameters=%s' % (self.name, pickle.dumps(self.hyperparameters))

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

    @staticmethod
    def hyperparameter_names():
        return []

    def get_status(self):
        '''return a printable string describing the status of the learner'''
        if self.count == 0:
            return 'Not Started'
        else:
            return '%s: Hyperparameters: %s, Updates: %d, Av. loss: %f, Av. gradient norm: %f' % \
            (self.name, str(self.hyperparameters), self.count, self.total_loss/(self.count), \
                self.total_gradient_norm/self.count)

def name_and_parameters_from_repr(repr_string):
    '''extracts the learner name and hyperparameter
    settings from the output of __repr__'''
    name = repr_string.split(';parameters=')[0]
    hyperparameters = pickle.loads(repr_string.split(';parameters=')[1])
    return name, hyperparameters

def dataset_name_from_repr(repr_string):
    '''extracts dataset name from repr string'''
    return re.search(r'Dataset\(name=(.*)\)', repr_string).group(0)

class LazyDataset(object):
    '''Behaves like an object of class Dataset, but
    doesn't load data until accessed'''

    def __init__(self, name, loader):
        self.name = name
        self.loader = loader
        self.dataset = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Dataset(name=%s)' % (self.name)

    def __getattr__(self, name):
        if name == 'name':
            return self.name

        if self.dataset is None:
            print 'Loading Dataset ' + self.name + '...'
            self.dataset = self.loader()
            print 'Loaded Dataset '+self.name
        return getattr(self.dataset, name)

def permute_dataset(feature_vectors, labels):
    '''randomly permutes a dataset'''
    dataset_size = np.shape(feature_vectors)[0]
    permutation = np.random.permutation(dataset_size)
    return feature_vectors[permutation], labels[permutation]

class Dataset(object):
    '''Stores a dataset and provides access via loss functions'''
    def __init__(self, name, feature_vectors, labels, problem_type='regression', \
            permute=False, num_classes=None, process_labels=False):
        self.name = name
        self.problem_type = problem_type
        self.num_classes = num_classes
        assert self.problem_type == 'regression' or self.problem_type == 'classification'

        if self.problem_type == 'regression':
            self.loss_func = l2_loss
        else:
            self.loss_func = multiclass_hinge_loss

        self.feature_vectors = feature_vectors
        self.labels = labels
        self.dataset_size = np.shape(self.feature_vectors)[0]

        if permute:
            self.feature_vectors, self.labels = permute_dataset(self.feature_vectors, self.labels)

        if self.problem_type == 'classification' and process_labels:
            relabeling = {}
            count = 0
            for class_index in np.unique(self.labels):
                relabeling[class_index] = count
                count += 1
            for example_index in xrange(len(self.labels)):
                self.labels[example_index] = relabeling[self.labels[example_index]]
        if self.problem_type == 'classification' and self.num_classes is None:
            self.num_classes = np.int(np.max(self.labels)) + 1

        self.shape = (self.num_classes, np.shape(self.feature_vectors)[1])

        if self.problem_type == 'regression':
            self.shape = (1, np.shape(self.feature_vectors)[1])
            self.num_classes = None

        self.current_index = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Dataset(name=%s)' % (self.name)

    def get_example(self):
        '''gets the next example in the dataset, looping to beginning if needed'''
        feature_vector = self.feature_vectors[self.current_index]
        label = self.labels[self.current_index]
        self.current_index = (self.current_index+1) % (self.dataset_size)

        return feature_vector, label

    def examples(self):
        '''yield each example in the dataset in turn'''
        for index in xrange(self.dataset_size):
            yield self.feature_vectors[index], self.labels[index]

    def get_infos(self):
        '''yields predict_info, get_loss_info pairs'''
        for index in xrange(self.dataset_size):
            yield self.feature_vectors[index], \
            lambda weights: self.loss_func(weights, self.feature_vectors[index], self.labels[index])


def load_libsvm_dataset(filename, name=None, problem_type='regression', \
        permute=False, num_classes=None):
    '''returns a lazy dataset for a given libsvm dataset file'''

    if name is None:
        name = os.path.basename(filename)

    return LazyDataset(name, lambda: Dataset(name, *load_svmlight_file(filename), \
        problem_type=problem_type, permute=permute, \
        num_classes=num_classes, process_labels=True))

def multiclass_hinge_loss(weights, feature_vector, label):
    '''computes multiclass hinge loss and its gradient

        weights: input prediction matrix
        feature_vector: input example to predict class for
        label: target class'''


    try:
        feature_vector = feature_vector.toarray()[0]
    except:
        pass
    prediction_scores = np.dot(weights, feature_vector)
    sorted_predictions = list(np.argsort(prediction_scores))
    best_prediction = sorted_predictions[-1]
    if best_prediction != label:
        true_loss = 1.0
    else:
        true_loss = 0.0
    try:
        sorted_predictions.remove(int(label))
    except:
        print 'predictions: ', prediction_scores
        print 'weights:', weights
        print 'feature vector:', feature_vector
        print 'label: ', label
        raise
    second_best_prediction = int(sorted_predictions[-1])
    label = int(label)

    gradient = np.zeros(np.shape(weights))
    hinge_loss = max(0.0, \
        1.0 + prediction_scores[second_best_prediction] - prediction_scores[label])
    if hinge_loss != 0:
        gradient[second_best_prediction] = feature_vector
        gradient[label] = -feature_vector
    return {'loss': true_loss, 'gradient': gradient, 'hinge_loss': hinge_loss}


def l2_loss(weights, feature_vector, label):
    '''computes squared error (w*x-y)^2 and its gradient with respect to w'''
    try:
        feature_vector = feature_vector.toarray()[0]
    except:
        pass

    prediction = feature_vector.dot(weights.T).flatten()[0]
    loss = (0.5*(prediction-label)**2).flatten()[0]
    gradient = (prediction-label)*feature_vector
    return {'loss': loss, 'gradient': gradient}



def run_learner(learner, dataset, status_interval=30):
    '''run a learner on a dataset, printing status
    every status_interval seconds'''
    print 'Uncached result - running experiment'
    start_time = time.time()
    last_status_time = 0
    losses = []
    total_loss = 0
    for predict_info, get_loss_info in dataset.get_infos():
        loss_info = get_loss_info(learner.predict(predict_info))
        losses.append(loss_info['loss'])
        total_loss += loss_info['loss']

        learner.update(loss_info)

        if time.time() > last_status_time + status_interval:
            last_status_time = time.time()
            print "%s, time elapsed: %d\r" % (learner.get_status(), last_status_time-start_time),
            sys.stdout.flush()
    print "%s, time elapsed: %d\r" % (learner.get_status(), last_status_time-start_time),
    sys.stdout.flush()
    print '\nDone!'
    return {'learner': learner.name, \
            'losses': losses, \
            'total_loss': total_loss, \
            'hyperparameters': learner.hyperparameters}


def search_hyperparameters(learner_factory, dataset, search_list):
    '''
    tries many hyperparameter settings for a learner on a dataset.
    search_dict is an iterable collection of hyperparameter dicts to input
    into learner.
    '''

    cachified_run_learner = cachelog.cachify(run_learner)

    for hyperparameters in search_list:
        cachified_run_learner(learner_factory(dataset.shape, hyperparameters), dataset)

def generate_default_search_list(learner_factory):
    '''
    yields all possible hyperparameter settings in a default
    logarithmically spaced grid.
    '''
    keys = learner_factory.hyperparameter_names()
    default_settings = np.power(10, np.arange(-5, 4.5, 0.5))
    current_indices = {key: 0 for key in keys}
    total_count = len(default_settings)**len(keys)
    count = 1
    yield {key: default_settings[current_indices[key]] for key in keys}
    while count < total_count:
        for key in keys:
            current_indices[key] = (current_indices[key] + 1) % len(default_settings)
            if current_indices[key] != 0:
                break
        yield {key: default_settings[current_indices[key]] for key in keys}
        count += 1

def run_all_datasets(directory, problem_type, learner_factories):
    '''
    given a directoy of libsvm datasets and some learners, runs hyperparameter searches
    on all the learners on all the datasets in the directory.
    '''
    filenames = [item for item in os.listdir(directory) \
        if os.path.isfile(os.path.join(directory, item))]

    try:
        filenames.remove('.')
        filenames.remove('..')
    except:
        pass

    for filename in filenames:
        dataset = load_libsvm_dataset(os.path.join(directory, filename), problem_type=problem_type, permute=False)
        for learner_factory in learner_factories:
            search_list = generate_default_search_list(learner_factory)
            search_hyperparameters(learner_factory, dataset, search_list)
