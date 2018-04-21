#!/usr/bin/env python3

def sma(data, key, n, field='Close'): # accurate since the n-th day
    sum = 0
    for i in range(n):
        sum += data[i][field]
        data[i][key] = sum / (i+1)
    for i in range(n, len(data)):
        sum += data[i][field] - data[i-n][field]
        data[i][key] = sum / n

def SMA5(data, key): # accurate since the 5-th day
    sma(data, key, 5)

def SMA10(data, key): # accurate since the 10-th day
    sma(data, key, 10)

def SMA20(data, key): # accurate since the 20-th day
    sma(data, key, 20)

def SMA60(data, key): # accurate since the 60-th day
    sma(data, key, 60)

# vi:et:sw=4:ts=4
