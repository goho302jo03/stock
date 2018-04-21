#!/usr/bin/env python3

import json
import model
import os
import xgboost
from pprint import pprint
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

codes = ['0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059',
         '006203', '006208', '00690', '00692',
         '00701', '00713']

if '__main__' == __name__:
    products_path = './twse_json'
    data = model.get_products(products_path)
    dataset = model.build_data(data, '20040501', '20180331')
    test_data = model.build_data(data, '20180401', '20180419')
    clfs = model.build_model(dataset, codes, '20040501', '20180331')
    for code in clfs:
        X = np.array(dataset[code]['X'])
        y = np.array(dataset[code]['y'])
        clfs[code].fit(X, y)
        pred = clfs[code].predict(X)
        rmse = sqrt(mean_squared_error(y, pred))
        print('{} \'s rmse: {}'.format(code, rmse))
# vi:et:sw=4:ts=4
