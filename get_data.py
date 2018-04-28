import numpy as np
import os, re, json, pprint

# 006201, 00690, 00692, 00701, 00713
codes = ['0050',   '0051',   '0052',   '0053',  '0054',  '0055',  '0056', '0057', '0058', '0059', 
         '006203', '006204', '006208' ]
selected_features = ['adj_close', 'close', 'high', 'low', 'open', 'volume']
dates = '201[3-7]-0[2-6]-.*\.json'
path = './json'
files = []

def dump_data():
    data = {}
    for filename in os.listdir(path):
        if re.match(dates, filename): files.append(filename)
    for file in files:
        day = json.load(open(path+'/'+file, 'r'))
        date = file.replace('.json', '')
        data[date] = {}
        for code in codes:
            data[date][code] = day[code]
    print(len(data.keys()))
    return data

def sort_data(path):
    datas = np.zeros((490, 14), dtype=object)
    with open(path, 'r') as f:
        file = json.load(f)

    print(len(file.keys()))
    for i, date in enumerate(file):
        datas[i][0] = int(date.replace('-', ''))
        for j, code in enumerate(codes):
            tmp = np.empty(0)
            for feature in selected_features:
                if file[date][code][feature] == 'NULL':
                    tmp = np.append(tmp, 0)
                else:
                    tmp = np.append(tmp, file[date][code][feature])
            datas[i][j+1] = tmp
    datas =  sorted(datas, key=lambda x:x[0])
    for code in range(13):
        x = np.empty((0, 30))
        y = np.empty(0)
        for i in range(90):
            tmp = np.empty(0)
            for j in range(5):
                tmp = np.append(tmp, datas[i+j][1+code])
            tmp = np.reshape(tmp, (-1, 30))
            x = np.vstack((x, tmp))
            y = np.append(y, datas[i+5][1+code][0])
        for i in range(95, 190):
            tmp = np.empty(0)
            for j in range(5):
                tmp = np.append(tmp, datas[i+j][1+code])
            tmp = np.reshape(tmp, (-1, 30))
            x = np.vstack((x, tmp))
            y = np.append(y, datas[i+5][1+code][0])
        for i in range(195, 286):
            tmp = np.empty(0)
            for j in range(5):
                tmp = np.append(tmp, datas[i+j][1+code])
            tmp = np.reshape(tmp, (-1, 30))
            x = np.vstack((x, tmp))
            y = np.append(y, datas[i+5][1+code][0])
        for i in range(291, 383):
            tmp = np.empty(0)
            for j in range(5):
                tmp = np.append(tmp, datas[i+j][1+code])
            tmp = np.reshape(tmp, (-1, 30))
            x = np.vstack((x, tmp))
            y = np.append(y, datas[i+5][1+code][0])
        for i in range(388, 485):
            tmp = np.empty(0)
            for j in range(5):
                tmp = np.append(tmp, datas[i+j][1+code])
            tmp = np.reshape(tmp, (-1, 30))
            x = np.vstack((x, tmp))
            y = np.append(y, datas[i+5][1+code][0])
        print(y.shape)
        np.save('./data/%s_x' %codes[code], x)
        np.save('./data/%s_y' %codes[code], y)

if '__main__' == __name__:
    # with open('data.json', 'w') as f:
    #     data = dump_data()
    #     json.dump(data, f, indent=4, sort_keys=True)
    sort_data('./data.json')
