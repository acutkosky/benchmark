'''
analyze stored data
'''

import pandas as pd

import benchmark as bm

def min_or_first(value1, value2):
    ''' if second argument is None, returns first argument. Else returns
    the minimum of its arguments.'''
    if value2 is None:
        return value1
    else:
        return min(value1, value2)

def get_dataframe_for_dataset(dataset_name, learners_to_hyperparameter_getters):
    '''
    gets a dataframe containing the average loss of each learner on a given dataset as a function
    of hyperparameter setting. The hyperparameter used is given as the value keyed
    by the learner's name in learners_to_hyperparameters.'''

    experiments = bm.extract_all_for_dataset(dataset_name)

    group_by_learners = {}
    for experiment in experiments:
        learner = experiment['learner']
        if learner in learners_to_hyperparameter_getters:
            if learner not in group_by_learners:
                group_by_learners[learner] = {}
            hyperparameter_getter = learners_to_hyperparameter_getters[learner]
            if isinstance(hyperparameter_getter, basestring):
                hyperparameter_name = hyperparameter_getter
                hyperparameter_getter = \
                    lambda _hyperparameters: _hyperparameters[hyperparameter_name]
            hyperparameters = experiment['hyperparameters']
            hyperparameter_setting = hyperparameter_getter(hyperparameters)
            average_loss = experiment['average_loss']
            group_by_learners[learner][hyperparameter_setting] = \
                min_or_first(average_loss, group_by_learners[learner].get(hyperparameter_setting))
    df = pd.DataFrame({learner: pd.Series(group_by_learners[learner]) \
        for learner in group_by_learners})
    df.index.name = 'hyperparameter setting'
    return df.interpolate(method='values')

def plot_dataset(dataset_name, learners_to_hyperparameters):
    '''
    plots average loss of each learner on a given dataset.
    '''
    df = get_dataframe_for_dataset(dataset_name, learners_to_hyperparameters)
    df.plot(title=dataset_name, logx=True, logy=True).set_ylabel('average loss')
