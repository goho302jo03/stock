#!/usr/bin/env python3

from bisect import bisect_left, bisect_right
import json
import numpy as np
from os import listdir
from re import sub
from pprint import pprint
import feature
import xgboost as xgb
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

codes = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059',
         '006203', '006208', '00690', '00692',
         '00701', '00713']
exception = ['', 'NULL', None]
# codes.append('006201', '006204')

def build_model(dataset, codes, start_date, end_date, re_fit=False): # {{{
    clf_list = {}
    for code in codes:
        X = np.array(dataset[code]['X'])
        y = np.array(dataset[code]['y'])
        clf = xgb.XGBRegressor()
        clf.fit(X, y)
        clf_list[code] = clf
    return clf_list
    # }}}


def build_data(products, start_date, end_date): # {{{
    selected_features = ['close', 'adj_close', 'open', 'low', 'high', 'volume']
    #start_index = 0
    #end_index = 67
    data = {}
    for code in codes:
        data[code] = {'X': [], 'y': []}
        dates = [v['date'] for v in products[code]]
        start_index = date_to_index(dates, start_date)
        end_index = date_to_index(dates, end_date)
        data_X = build_X(products, start_index, end_index, selected_features, 3)
        data_y = build_y(products, start_index, end_index)
        for X, y in list(zip(data_X[code], data_y[code]['price'])):
            if y is not None:
                data[code]['X'].append(X)
                data[code]['y'].append(y)
    return data
    # }}}

def build_X(products, start_index, end_index, selected_features, feature_days): # {{{
    data = {}
    for code in codes:
        X = []
        for i in range(start_index, end_index+1):
            X.append([])
            for j in range(i - feature_days, i):
                X[-1].extend([products[code][j].get(v) for v in selected_features])

        X_float = [[ None if ele in exception else float(ele) for ele in sample ] for sample in X]
        data[code] = X_float
    return data
    # }}}

def build_y(products, start_index, end_index): # {{{

    def get_dir(product, i):
        if product[i] >= product[i-1]: return 1
        if product[i] < product[i-1]: return -1

    data = {}
    for code in codes:
        data[code] = { 'dir': [], 'price': [] }
        product = [v.get('close') for v in products[code]]
        for i in range(start_index, end_index+1):
            data[code]['price'].append(None if product[i] in exception else float(product[i]))
            data[code]['dir'].append(None if product[i] or product[i-1] in exception else get_dir(product[i]))
    return data
    # }}}

def date_to_index(dates, date, greater=True): # {{{
    if (greater):
        return bisect_left(dates, date)
    else:
        return bisect_right(dates, date) - 1
    # }}}

def get_products(path): # {{{
    data = {}
    for code in codes:
        data[code] = []
    for filename in sorted(listdir(path)):
        day = json.load(open('%s/%s' % (path, filename)))
        date = sub(r'-|.json$', '', filename)
        for code in codes:
            data[code].append(day.get(code, {}))
            data[code][-1]['date'] = date
    return data
    # }}}

# vi:et:sw=4:ts=4
