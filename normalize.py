import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 有指數: '0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P'
# 沒指數: '0054', '0055'

# features = ['Date', 'Open', 'High', 'Low', 'Close', 'SMA5', 'SMA10', 'SMA20', 'SMA60', 'Volume', 'MA5', 'MA10', 'DIF', 'MACD9', 'OSC', 'K', 'D', 'RSI 6', 'RSI 12', 'RSV9']
codes = ['0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P', '0054', '0055']

feature_days = 5

def Stander(tr, te, code):
    standardscaler_x = StandardScaler()
    standardscaler_y = StandardScaler()
    standardscaler_x.fit(tr[:, 10:])
    standardscaler_y.fit(tr[:, (1,3,5,7,9)])
    tr[:, 10:]         = standardscaler_x.transform(tr[:, 10:])
    tr[:, (1,3,5,7,9)] = standardscaler_y.transform(tr[:, (1,3,5,7,9)])
    te[:, 10:]         = standardscaler_x.transform(te[:, 10:])
    te[:, (1,3,5,7,9)] = standardscaler_y.transform(te[:, (1,3,5,7,9)])

    np.save('./data/day_%d/StandardScaler/tr_%s' %(feature_days, code), tr)
    np.save('./data/day_%d/StandardScaler/te_%s' %(feature_days, code), te)
    print('finish standard_scaler: %s, %d, %d' %(code, len(tr), len(te)))

def MinMax(tr, te, code):
    minmaxscaler_x = MinMaxScaler()
    minmaxscaler_y = MinMaxScaler()
    minmaxscaler_x.fit(tr[:, 10:])
    minmaxscaler_y.fit(tr[:, (1,3,5,7,9)])
    tr[:, 10:]         = minmaxscaler_x.transform(tr[:, 10:])
    tr[:, (1,3,5,7,9)] = minmaxscaler_y.transform(tr[:, (1,3,5,7,9)])
    te[:, 10:]         = minmaxscaler_x.transform(te[:, 10:])
    te[:, (1,3,5,7,9)] = minmaxscaler_y.transform(te[:, (1,3,5,7,9)])

    np.save('./data/day_%d/MinMaxScaler/tr_%s' %(feature_days, code), tr)
    np.save('./data/day_%d/MinMaxScaler/te_%s' %(feature_days, code), te)
    print('finish MinMaxScaler: %s, %d, %d' %(code, len(tr), len(te)))

for code in codes:

    tr = np.load('./data/day_%d/non_normalize/tr_%s.npy' %(feature_days, code))
    te = np.load('./data/day_%d/non_normalize/te_%s.npy' %(feature_days, code))

    Stander(tr, te, code)
    MinMax(tr, te, code)

