'''
manage database of experiments
'''
import sqlite3
import time
import json

DEFAULT_NAME = 'experiments.db'
CONN = None
CURSOR = None

TABLE_NAME = "experiments"
FIELDS = { \
        'time': 'REAL', \
        'dataset': 'TEXT', \
        'learner': 'TEXT', \
        'total_loss': 'REAL', \
        'iterations': 'INTEGER', \
        'losses': 'JSON', \
        'hyperparameters': 'JSON', \
        'extra_data': 'JSON'}

JSON_FIELDS = [_ for _ in FIELDS if FIELDS[_] == 'JSON']

for _ in JSON_FIELDS:
    FIELDS[_] = 'TEXT'

SYMBOL_PLACEHOLDER = ', '.join(['?' for _ in FIELDS])

def dict_to_field_string(to_convert, join_str=',', op_str=''):
    '''
    converts a dictionary D into a string of the form
    "k1 op_str D[k1] join_str k2 op_str D[k2], ..."
    where k1, k2 re the keys.
    '''
    return (' %s ' % (join_str)).join(["%s %s %s" % (key, op_str, to_convert[key]) \
        for key in to_convert])

def initialize(db_name=None):
    '''initialize the DB'''
    if db_name is None:
        db_name = DEFAULT_NAME

    global CONN
    global CURSOR

    CONN = sqlite3.connect(db_name)
    CURSOR = CONN.cursor()

    create_string = dict_to_field_string(FIELDS)
    query = "CREATE TABLE IF NOT EXISTS %s (%s)" % (TABLE_NAME, create_string)

    CURSOR.execute(query)
    CONN.commit()

def add_experiment(experiment_data):
    '''
    add an experiment's data into the db
    '''
    field_keys = FIELDS.keys()
    field_keys = [_ for _ in field_keys if _ in experiment_data]
    insert_item = [json.dumps(experiment_data[_]) if _ in JSON_FIELDS else \
        experiment_data[_] for _ in field_keys]

    if 'time' not in field_keys:
        insert_item.append(time.time())
        field_keys.append('time')

    if 'extra_data' not in field_keys:
        insert_item.append(json.dumps({}))
        field_keys.append('extra_data')

    insert_item = [tuple(insert_item)]
    
    field_string = ', '.join(["'%s'" % (_) for _ in field_keys])

    query = "INSERT INTO experiments (%s) VALUES (%s)" % (field_string, SYMBOL_PLACEHOLDER)

    CURSOR.executemany(query, insert_item)
    CONN.commit()

def select_all():
    return CURSOR.execute("select * from experiments").fetchall()

def recover_experiment(where, select=None):
    '''
    recovers selected fields from an experiment given a dict of
    fields to search for.
    '''
    if select is None:
        select = FIELDS.keys()

    select_symbol_placeholder = ', '.join(select)
    where_symbol_placeholder = ' AND '.join(["%s=?" % (_) for _ in where])

    symbol_values = where.values()

    query = "SELECT %s FROM %s WHERE %s ORDER BY time DESC" % \
        (select_symbol_placeholder, TABLE_NAME, where_symbol_placeholder)

    results_tuples = CURSOR.execute(query, symbol_values).fetchall()

    results_dicts = [dict(zip(select, list(results))) for results in results_tuples]

    for item in results_dicts:
        for json_field in JSON_FIELDS:
            if json_field in item:
                item[json_field] = json.loads(item[json_field])

    return results_dicts
