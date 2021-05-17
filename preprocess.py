import numpy as np


symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '+']


def encode(x):
    encoded_data = []
    for operations in x:
        int_data = [symbols.index(value) for value in operations]
        encoded_data.append(int_data)
    return encoded_data


def preprocess(x):
    one_hot_data = []
    data = encode(x)
    for value in data:
        temp = []
        for i, j in enumerate(value):
            zeros = np.zeros(len(symbols))
            zeros[j] = 1
            temp.append(zeros)
        one_hot_data.append(temp)
    x = np.array(one_hot_data)
    return x


def result_preprocess(x):
    int_data = dict((i, j) for i, j in enumerate(symbols))
    temp = []
    for value in x:
        string = int_data[np.argmax(value)]
        temp.append(string)
    return ''.join(temp)



