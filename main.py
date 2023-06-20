import numpy as np
import asyncio


# TODO: optimize function to remove three nested arrays
async def trainTrueVals(trueArr, percentages):
    uniqueTrue, countsTrue = np.unique(trueArr, return_counts=True)
    freqDictTrue = dict(zip(uniqueTrue, countsTrue))
    totalSzTrue = len(uniqueTrue)
    arMaxTrue = np.max(trueArr)
    arMinTrue = np.min(trueArr)

    for sz in range(2, totalSzTrue + 1):
        for starting in range(arMinTrue - totalSzTrue, arMaxTrue + 1):
            sum = 0
            for i in range(starting, starting + sz + 1):
                sum += freqDictTrue[i] if i in uniqueTrue else 0
            for i in range(starting, starting + sz + 1):
                # initially only storing the numertor;
                if i not in percentages:
                    percentages[i] = 0
                percentages[i] += sum


# TODO: optimize function to remove three nested arrays
async def trainFalseVals(falseArr, percentages):
    uniqueFalse, countsFalse = np.unique(falseArr, return_counts=True)
    freqDictFalse = dict(zip(uniqueFalse, countsFalse))
    totalSzFalse = len(uniqueFalse)
    arMaxFalse = np.max(falseArr)
    arMinFalse = np.min(falseArr)

    for sz in range(2, totalSzFalse + 1):
        for starting in range(arMinFalse - totalSzFalse, arMaxFalse + 1):
            sum = 0
            for i in range(starting, starting + sz + 1):
                sum += freqDictFalse[i] if i in uniqueFalse else 0
            for i in range(starting, starting + sz + 1):
                # initially only storing the numertor;
                if i not in percentages:
                    percentages[i] = 0
                percentages[i] -= sum


async def train(trueArr, falseArr):
    """
    trueArr: an 1D numpy array of integers only (containing the true value set), for example in case of titanic death data, give an
    array of all deaths age [20, 21, 21, 21, 50, ...] to predict for any given age the chance of its death.

    falseArr: Similar to above but containing the false values
    """

    percentagesRaw = {}
    trainTrueValsTask = asyncio.create_task(trainTrueVals(trueArr, percentagesRaw))
    await trainTrueValsTask

    trainFalseValsTask = asyncio.create_task(trainFalseVals(falseArr, percentagesRaw))
    await trainFalseValsTask

    percentagesRawMax = np.max(np.array(list(percentagesRaw.values())))
    for key in percentagesRaw.keys():
        percentagesRaw[key] = (float(percentagesRaw[key]) * 100) / float(
            percentagesRawMax
        )

    return percentagesRaw


print(asyncio.run(train([1, 2, 3, 4, 4, 4, 5, 10], [0, 1, 4, 5, 9, 10, 11])))
