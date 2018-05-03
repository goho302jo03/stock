import numpy as np
import re
from operator import itemgetter
from sklearn.preprocessing import StandardScaler


# features = ['Date', 'Open', 'High', 'Low', 'Close', 'SMA5', 'SMA10', 'SMA20', 'SMA60', 'Volume', 'MA5', 'MA10', 'DIF', 'MACD9', 'OSC', 'K', 'D', 'RSI 6', 'RSI 12', 'RSV9']
# 有指數: '0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P'
# 沒指數: '0054', '0055'

# 2010前沒資料(begin要設為None)
codes = ['00690P', '00692P', '00701P', '00713P']
# 2010前有資料
# codes = ['0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '0054', '0055']

select_features = [1, 2, 3, 4]
feature_days = 5
getter = itemgetter(*select_features)

def build_data(begin=None, end=None, t='tr'):

    for code in codes:

        with open('./products/%s.csv' %code, 'r') as f:
            data = f.read().split('\n')

        flag = 0
        data = data[1:-1]
        etf = np.empty((0, feature_days*len(select_features)+10))

        for i in range(len(data)):
            data[i] = data[i].split(',')

        for day in range(len(data)-feature_days-4):
            if begin and flag == 0 and re.match(begin, data[day][0]): flag = 1
            if begin == None: flag = 1
            if flag == 0: continue
            tmp = np.empty(0)

            for after_day in range(5):
                # print(data[day + feature_days + after_day])
                ud = float(data[day + feature_days + after_day][4]) - float(data[day + feature_days + after_day - 1][4])
                tmp = np.append(tmp, (1 if ud >= 0 else 0))
                tmp = np.append(tmp, ud/float(data[day + feature_days + after_day - 1][4]))

            for feature_day in range(feature_days):
                tmp = np.append(tmp, getter(data[day + feature_day]))
            tmp = np.reshape(tmp.astype(float), (1, feature_days*len(select_features)+10))
            etf = np.vstack((etf, tmp))

            if end and re.match(end, data[day][0]): break

        np.save('./data/day_%d/non_normalize/%s_%s' %(feature_days, t, code), etf)
        print('finish %s %s %d' %(t, code, len(etf)))

build_data(end='2017/12/29', t='tr')
# build_data(begin='2010/1/2', end='2017/12/29', t='tr')
build_data(begin='2018/1/2', t='te')
# build_data(begin='2018/2/27')
# build_data()
