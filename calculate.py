import numpy as np
import json

# 有指數: '0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '0059P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P'
codes = ['0050P', '0051P', '0052P', '0053P', '0056P', '0057P', '0058P', '006201P', '006203P', '006204P', '006208P', '00690P', '00692P', '00701P', '00713P']
# 沒指數: '0054', '0055'
# codes = ['0059P']

index_price_dict = {}

for code in codes:

    price_code = code.replace('P', '')

    with open('./products/%s.csv' %code, 'r') as f:
        index = f.read().split('\n')
        index = index[1:-1]

    with open('./products/%s.csv' %price_code, 'r') as f:
        price = f.read().split('\n')
        price = price[1:-1]

    for i in range(len(index)):
        index[i] = index[i].split(',')

    for i in range(len(price)):
        price[i] = price[i].split(',')

    price = np.array(price)
    index = np.array(index)

    index_mean = np.mean(index[:, 4].astype(float))
    price_mean = np.mean(price[:, 4].astype(float))

    ratio = price_mean/index_mean

    print(code, ratio, price_mean, index_mean)

    index_price_dict[code] = {
        'index_mean': index_mean,
        'price_mean': price_mean,
        'ratio': ratio
    }

with open('index_price_dict.json', 'w') as outfile:
    json.dump(index_price_dict, outfile, indent=4)

