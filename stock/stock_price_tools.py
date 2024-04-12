import pandas as pd
import numpy as np


def calculate_rsi(data, period=14):
    '''
    :param data: stock price data
    :param period: period of dates
    :return: relative strength index (RSI)
    :rtype: float
    '''
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    '''
    Calculates the convergence divergence moving average

    :param data: stock price data
    :param short_period: no. period used to calculate shorter exponential moving average (EMA)
    :param long_period: no. period used to calculate longer exponential moving average (EMA)
    :param signal_period: no. period to calculate the signal line
    :return:macd
    '''
    short_ema = data['Close'].ewm(span=short_period, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_period, min_periods=1).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()

    return macd_line, signal_line


def calculate_sma(data, window):
    '''
    Calculates the single moving average

    :param data: stock price data of one ticker
    :param window: size of the sliding window (no. of days)
    :return:
    '''
    sma = data['Close'].rolling(window=window).mean()
    return sma
