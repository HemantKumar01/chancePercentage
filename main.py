import numpy as np
import pandas as pd
import pickle


def getFreqDict(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


def subtractFreqDicts(dict1, dict2):
    for i in dict2.keys():
        if i not in dict1.keys():
            dict1[i] = 0
        dict1[i] -= dict2[i]
    return dict1


# TODO: optimize time complexity of fitData function
def fitData(freqDict, model):
    unique = list(freqDict.keys())
    totalSz = len(unique)
    arMax = np.max(unique)
    arMin = np.min(unique)

    for i in range(arMin - totalSz, arMax + totalSz + 1):
        if i not in unique:
            freqDict[i] = 0

        for j in range(arMin - totalSz, arMax + totalSz + 1):
            if j not in model.keys():
                model[j] = 0

            a = np.sqrt((arMax + totalSz) - (arMin - totalSz)) / 2
            if (a**2 - (j - i) ** 2) < 0:
                continue
            model[j] += (freqDict[i] / a**2) * (a**2 - (j - i) ** 2)

    return model


def postProcess(model):
    maxVal = float(np.max(list(model.values())))
    for key in model.keys():
        model[key] = float(model[key]) * 100.0 / maxVal
    return model


def train(trueArr, falseArr):
    """
    trueArr: an 1D numpy array of integers only (containing the true value set), for example in case of titanic death data, give an array of all deaths age [20, 21, 21, 21, 50, ...] to predict for any given age the chance of its death.

    falseArr: Similar to above but containing the false values
    """
    model = {}

    freqDict = subtractFreqDicts(getFreqDict(trueArr), getFreqDict(falseArr))
    model = fitData(freqDict, model)

    model = postProcess(model)
    return model


print(train([1, 2, 3, 3, 3, 4, 4, 9, 12], [1]))
