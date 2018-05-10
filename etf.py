import numpy as np
import json, datetime
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

class Feature():

    def __init__(self, code, feature_day, state, selected_feature, scaler):

        self.code = code
        self.feature_day = feature_day
        self.state = state
        self.selected_feature = selected_feature
        self.scaler = scaler
        self.etf = None

    def build_data(self, begin, end, tr_data=None):

        with open('../../products/%s.csv' %self.code, 'r') as f:
            data = f.read().split('\n')[1: -1]

        # transfer yyyy/m/d to yyyymmdd
        for i in range(len(data)):
            data[i] = data[i].split(',')
            tmp = data[i][0].split('/')
            if len(tmp[1]) == 1:
                tmp[1] = '0' + tmp[1]
            if len(tmp[2]) == 1:
                tmp[2] = '0' + tmp[2]
            data[i][0] = ''.join(tmp)

        # find data between begin and end
        data = np.array(data)
        data = data[np.where(data[:,0].astype(int) >= begin)]
        data = data[np.where(data[:,0].astype(int) <= end)]

        etf = self.build_xy(data)
        self.etf = etf

        if self.scaler == 'NoScaler':
            return etf
        elif self.scaler == 'StandardScaler':
            return self.__standard_scaler(etf, tr_data)
        elif self.scaler == 'MinMaxScaler':
            return self.__minmax_scaler(etf, tr_data)
        elif self.scaler == 'PerStandardScaler':
            return self.__per_standard_scaler(etf, tr_data)
        elif self.scaler == 'PerMinMaxScaler':
            return self.__per_minmax_scaler(etf, tr_data)

    def build_xy(self, data):

        getter = itemgetter(*self.selected_feature)

        # generate x and y
        etf = np.empty((0, self.feature_day*len(self.selected_feature)+20))
        for day in range(len(data) - self.feature_day - 4):
            tmp = np.empty(0)
            count = 0
            # generate y
            for after_day in range(5):
                ud = float(data[day + self.feature_day + after_day][4]) - float(data[day + self.feature_day + after_day - 1][4])
                if  abs(ud/float(data[day + self.feature_day + after_day - 1][4]))> 0.03:
                    count = 1
                    break
                tmp = np.append(tmp, data[day + self.feature_day + after_day][0])
                tmp = np.append(tmp, (1 if ud >= 0 else 0))
                tmp = np.append(tmp, ud/float(data[day + self.feature_day + after_day - 1][4]))
                tmp = np.append(tmp, data[day + self.feature_day + after_day][4])
            # delete outlier of ud rate
            if self.state == 'delete_outlier' and count > 0: continue
            # generate x
            for i in (range(self.feature_day)):
                tmp = np.append(tmp, getter(data[day + i]))

            tmp = np.reshape(tmp.astype(float), (1, self.feature_day*len(self.selected_feature)+20))
            etf = np.vstack((etf, tmp))

        return etf

    def __per_standard_scaler(self, te, tr=None):

        if tr is None: tr = te

        te[:, 20:] = scale(te[:, 20:], axis=1)
        standardscaler_y = StandardScaler()
        standardscaler_y.fit(tr[:, (2,6,10,14,18)])
        for i in range(len(te)):
            te[i, (2,6,10,14,18)] = standardscaler_y.transform(np.reshape(te[i, (2,6,10,14,18)], (1, -1)))

        return te

    def __per_minmax_scaler(self, te, tr=None):

        if tr is None: tr = te

        te[:, 20:] = scale(te[:, 20:], axis=1)
        minmaxscaler_y = MinMaxScaler()
        minmaxscaler_y.fit(tr[:, (2,6,10,14,18)])
        for i in range(len(te)):
            te[i, (2,6,10,14,18)] = minmaxscaler_y.transform(np.reshape(te[i, (2,6,10,14,18)], (1, -1)))

        return te

    def __standard_scaler(self, te, tr=None):

        if tr is None: tr = te

        standardscaler_x = StandardScaler()
        standardscaler_y = StandardScaler()
        standardscaler_x.fit(tr[:, 20:])
        standardscaler_y.fit(tr[:, (2,6,10,14,18)])
        for i in range(len(te)):
            te[i, 20:]            = standardscaler_x.transform(np.reshape(te[i, 20:], (1, -1)))
            te[i, (2,6,10,14,18)] = standardscaler_y.transform(np.reshape(te[i, (2,6,10,14,18)], (1, -1)))

        return te

    def __minmax_scaler(self, te, tr=None):

        if tr is None: tr = te

        minmaxscaler_x = MinMaxScaler()
        minmaxscaler_y = MinMaxScaler()
        minmaxscaler_x.fit(tr[:, 20:])
        minmaxscaler_y.fit(tr[:, (2,6,10,14,18)])
        for i in range(len(te)):
            te[i, 20:]            = minmaxscaler_x.transform(np.reshape(te[i, 20:], (1, -1)))
            te[i, (2,6,10,14,18)] = minmaxscaler_y.transform(np.reshape(te[i, (2,6,10,14,18)], (1, -1)))

        return te

    def deScaler(self, data, y_pred):

        data = np.reshape(data, (-1, 1))
        scaler_value = np.reshape(y_pred, (-1, 1))

        if self.scaler == 'StandardScaler':
            standardscaler = StandardScaler()
            standardscaler.fit(data)
            new_y_pred = standardscaler.inverse_transform(scaler_value)

        elif self.scaler == 'MinMaxScaler':
            minmaxscaler = MinMaxScaler()
            minmaxscaler.fit(data)
            new_y_pred = minmaxscaler.inverse_transform(scaler_value)

        return new_y_pred

    def percentToIndex(self, y_pred, day_pred):

        data = self.etf

        if self.scaler != 'NoScaler':
            print('22')
            y_pred = self.deScaler(data[:, (day_pred*4-2)], y_pred)

        y_pred = np.reshape(y_pred, (-1))
        origin = data[:, (day_pred*4-1)] / (1+data[:, (day_pred*4-2)])
        pred_index = (1+y_pred) * origin

        return pred_index

    def indexToPrice(self, pred_index):

        with open('../../index_price_dict.json', 'r') as f:
            dict = json.load(f)

        ratio = dict[self.code]['ratio']

        return pred_index * ratio

    def percentToPrice(self, y_pred, day_pred):

        pred_index = self.percentToIndex(y_pred, day_pred)
        pred_answer = self.indexToPrice(pred_index)

        return pred_answer

    def dateToWeekday(self, data, day_pred):

        week_day = []
        for i in range(len(data)):
            date = data[i][day_pred*4-4]
            year = int(date//10000)
            month = int((date%10000)//100)
            day = int(date%100)
            week_day.append(datetime.datetime(year, month, day).strftime("%w") )

        return np.array(week_day)

    def accWeekday(self, data, y_pred, day_pred, show_date=1):

        week_day = self.dateToWeekday(data, day_pred)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        for i in range(1, 6):
            acc = 0
            tmp = np.where(week_day=='%d'%i)
            for j in tmp[0]:
                if show_date: print(data[j][0])
                acc += int(y_pred[j] == data[j][day_pred*4-3])
            print('星期%d : %f'%(i, acc/len(tmp[0])))

    # def output(path, ):
    #
    #     



