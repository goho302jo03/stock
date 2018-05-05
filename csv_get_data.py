import numpy as np
import re
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# features = ['Date', 'Open', 'High', 'Low', 'Close', 'SMA5', 'SMA10', 'SMA20', 'SMA60', 'Volume', 'MA5', 'MA10', 'DIF', 'MACD9', 'OSC', 'K', 'D', 'RSI 6', 'RSI 12', 'RSV9']
# 有指數: '0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P'
# 沒指數: '0054', '0055'


# 2010
# codes = ['0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '006201P', '006203P', '006204P', '006208P', '0054', '0055']

# full year
# codes = ['0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '006201P', '006203P', '006204P', '006208P', '0054', '0055', '00690P', '00692P', '00701P', '00713P']

# codes = ['0059P']
codes = ['0050P']

select_features = [1, 2, 3, 4, 9]
feature_days = [5, 15, 22, 30]
opt = {
    'state': ['origin', 'delete_outlier'],  # ['delete_outlier', 'origin']
    'interval': 'full',         # ['full', year_n]
}
getter = itemgetter(*select_features)

def build_data(begin=None, end=None, t='tr'):

    for state in opt['state']:

        for feature_day in feature_days:

            for code in codes:

                with open('./products/%s.csv' %code, 'r') as f:
                    data = f.read().split('\n')

                flag = 0
                data = data[1:-1]
                etf = np.empty((0, feature_day*len(select_features)+15))

                for i in range(len(data)):
                    data[i] = data[i].split(',')

                for day in range(len(data)-feature_day-4):
                    if begin and flag == 0 and re.match(begin, data[day][0]): flag = 1
                    if begin == None: flag = 1
                    if flag == 0: continue

                    tmp = np.empty(0)
                    count = 0

                    for after_day in range(5):
                        # print(data[day + feature_days + after_day])
                        ud = float(data[day + feature_day + after_day][4]) - float(data[day + feature_day + after_day - 1][4])
                        tmp = np.append(tmp, (1 if ud >= 0 else 0))
                        tmp = np.append(tmp, ud/float(data[day + feature_day + after_day - 1][4]))
                        tmp = np.append(tmp, data[day + feature_day + after_day][4])
                        if  abs(ud/float(data[day + feature_day + after_day - 1][4]))> 0.03:
                            count += 1
                            # print(data[day + feature_days + after_day - 1][0], ud/float(data[day + feature_days + after_day - 1][4]))

                    if state=='delete_outlier' and count > 0: continue # 去除outlier

                    for i in range(feature_day):
                        tmp = np.append(tmp, getter(data[day + i]))
                    tmp = np.reshape(tmp.astype(float), (1, feature_day*len(select_features)+15))
                    etf = np.vstack((etf, tmp))

                    if end and re.match(end, data[day][0]): break

                np.save('./data_2/%s/day%d/NoScaler/%s_%s_%s' %(state, feature_day, t, code, opt['interval']), etf)
                print('finish %s %s %d' %(t, code, len(etf)))

def analyze_ud(etf):

    print('-------------------------------')
    for i in range(5):
        ud_list = []
        for j in range(len(etf)):
            ud_list.append(abs(etf[j][i*2 + 1]))

        mean = np.mean(ud_list)
        std = np.std(ud_list)
        print('mean: %f, var: %f, 99.7 percents data will locate in %f - %f' %(mean, std, mean-3*std, mean+3*std))

def Stander(tr, te, code, feature_day, state):
    standardscaler_x = StandardScaler()
    standardscaler_y = StandardScaler()
    standardscaler_x.fit(tr[:, 15:])
    standardscaler_y.fit(tr[:, (1,4,7,10,13)])
    tr[:, 15:]         = standardscaler_x.transform(tr[:, 15:])
    tr[:, (1,4,7,10,13)] = standardscaler_y.transform(tr[:, (1,4,7,10,13)])
    te[:, 15:]         = standardscaler_x.transform(te[:, 15:])
    te[:, (1,4,7,10,13)] = standardscaler_y.transform(te[:, (1,4,7,10,13)])

    np.save('./data_2/%s/day%d/StandardScaler/tr_%s_%s' %(state, feature_day, code, opt['interval']), tr)
    np.save('./data_2/%s/day%d/StandardScaler/te_%s_%s' %(state, feature_day, code, opt['interval']), te)
    print('finish standard_scaler: %s, %d, %d' %(code, len(tr), len(te)))

def MinMax(tr, te, code, feature_day, state):
    minmaxscaler_x = MinMaxScaler()
    minmaxscaler_y = MinMaxScaler()
    minmaxscaler_x.fit(tr[:, 15:])
    minmaxscaler_y.fit(tr[:, (1,4,7,10,13)])
    tr[:, 15:]         = minmaxscaler_x.transform(tr[:, 15:])
    tr[:, (1,4,7,10,13)] = minmaxscaler_y.transform(tr[:, (1,4,7,10,13)])
    te[:, 15:]         = minmaxscaler_x.transform(te[:, 15:])
    te[:, (1,4,7,10,13)] = minmaxscaler_y.transform(te[:, (1,4,7,10,13)])

    np.save('./data_2/%s/day%d/MinMaxScaler/tr_%s_%s' %(state, feature_day, code, opt['interval']), tr)
    np.save('./data_2/%s/day%d/MinMaxScaler/te_%s_%s' %(state, feature_day, code, opt['interval']), te)
    print('finish MinMaxScaler: %s, %d, %d' %(code, len(tr), len(te)))
    # print(minmaxscaler_x.data_max_, minmaxscaler_x.data_min_, minmaxscaler_y.data_max_, minmaxscaler_y.data_min_)

if __name__ == '__main__':

    build_data(end='2017/12/29', t='tr')
    # build_data(begin='2010/1/2', end='2017/12/29', t='tr')
    build_data(begin='2018/1/2', t='te')
    # build_data(begin='2018/2/27')
    # build_data()

    for state in opt['state']:
        for code in codes:
            for feature_day in feature_days:
                tr = np.load('./data_2/%s/day%d/NoScaler/tr_%s_%s.npy' %(state, feature_day, code, opt['interval']))
                te = np.load('./data_2/%s/day%d/NoScaler/te_%s_%s.npy' %(state, feature_day, code, opt['interval']))

                Stander(tr, te, code, feature_day, state)
                MinMax(tr, te, code, feature_day, state)




