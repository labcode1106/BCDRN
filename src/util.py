# coding=utf-8

import math
import os
import numpy
import pandas
import datetime


def removeRepetitionInList(list1):
    list2 = []
    for i in list1:
        if not i in list2:
            list2.append(i)
    return list2


def calRepetitionCountInList(list1, list2):
    count = 0
    for i in list1:
        if i in list2:
            count += 1
    return count


def getProbationPeriod(probationPercent, fileLength):
    """Return the probationary period index."""
    return min(
        math.floor(probationPercent * fileLength),
        probationPercent * 5000)


def get_nth_parent_dir(n, path):
    """
    Return the Nth parent of `path` where the 0th parent is the direct parent
    directory.
    """
    parent = os.path.dirname(path)
    if n == 0:
        return parent

    return get_nth_parent_dir(n - 1, parent)


def convertResultsPathToDataPath(path):
    """
  @param path (string)  Path to dataset in the data directory.

  @return     (string)  Path to dataset result in the result directory.
  """
    path = path.split(os.path.sep)
    detector = path[0]
    path = path[1:]

    filename = path[-1]
    toRemove = detector + "_"

    i = filename.index(toRemove)
    filename = filename[:i] + filename[i + len(toRemove):]

    path[-1] = filename
    path = "/".join(path)

    return path


def calXOrForTwoList(list1, list2):
    list = []
    for (i1, i2) in zip(list1, list2):
        if i1 == i2:
            list.append(0)
        else:
            list.append(1)
    return list


def calAndForTwoList(list1, list2):
    list = []
    for (i1, i2) in zip(list1, list2):
        if i1 == 1 and i2 == 1:
            list.append(1)
        else:
            list.append(0)
    return list


def calOrForTwoList(list1, list2):
    list = []
    for (i1, i2) in zip(list1, list2):
        if i1 == 1 or i2 == 1:
            list.append(1)
        else:
            list.append(0)
    return list


def convertLabelsBasedOnDataReduction(labels, combinedNum):
    results = []
    hasAnomalyLabel = 0
    for i in range(0, len(labels)):
        if labels["label"][i] == 1:
            hasAnomalyLabel = 1
        if (i + 1) % combinedNum == 0:  # 遍历一组结束
            results.append((labels["timestamp"][i], hasAnomalyLabel))
            hasAnomalyLabel = 0
    out = pandas.DataFrame(data=results, columns=['timestamp', 'label'])
    return out


def convertLabelsBasedOnDataReduction_enhence(labels, combinedNum):
    results = []
    for i in range(combinedNum - 1, len(labels)):
        results.append((labels["timestamp"][i], labels["label"][i]))
    out = pandas.DataFrame(data=results, columns=['timestamp', 'label'])
    return out


def calConvoluteEncoding(finalEncoding, convWidth, poolingWidth, timeOfDayEncoderWidth):
    # print ("finalEncoding length before conv: %s" % len(finalEncoding))
    # print finalEncoding
    if not isinstance(finalEncoding, list):
        finalEncoding = finalEncoding.tolist()
    timeEncoding = finalEncoding[0:timeOfDayEncoderWidth]
    scalarEncoding = finalEncoding[timeOfDayEncoderWidth::]
    """ conv c==convWidth stride==1 """
    convList = []
    for i in range(0, len(scalarEncoding) - convWidth + 1):
        tmp = 0
        for j in range(0, convWidth):
            tmp += scalarEncoding[i + j]
        convList.append(tmp)

    """ maxpooling stride==poolingWidth """
    poolingList = []
    max = 0
    for i in range(0, len(convList)):
        if (i + 1) % poolingWidth == 0 or i == len(convList) - 1:
            poolingList.append(max)
            max = 0
        else:
            if convList[i] > max:
                max = convList[i]
    out = timeEncoding + poolingList


    # """ averagepooling stride==poolingWidth """
    # poolingList = []
    # sum = 0
    # for i in range(0, len(convList)):
    #     if (i + 1) % poolingWidth == 0 or i == len(convList) - 1:
    #         avg=int(float(sum)/poolingWidth+0.5)
    #         poolingList.append(avg)
    #         sum = 0
    #     else:
    #         sum = sum + convList[i]
    # out = timeEncoding + poolingList

    return out


def convertEncodingTo0And1(finalEncoding, timeOfDayEncoderWidth):
    timeEncoding = finalEncoding[0:timeOfDayEncoderWidth]
    poolingList = finalEncoding[timeOfDayEncoderWidth::]
    for i in range(0, len(poolingList)):
        if poolingList[i] > 1:
            poolingList[i] = 1
    out = timeEncoding + poolingList
    return out


def calAverageEncoding(finalEncoding, convWidth, timeOfDayEncoderWidth):
    """ flat convoluted values """
    timeEncoding = finalEncoding[0:timeOfDayEncoderWidth]
    poolingList = finalEncoding[timeOfDayEncoderWidth::]
    flattedList = []
    for i in range(0, len(poolingList)):
        # redundantVal=poolingList[i]-1
        # radius=1
        # while redundantVal>0:     # 将多余的值分摊到周围的bit
        #     if i+radius<len(poolingList) and poolingList[i + radius] == 0:
        #         poolingList[i + radius] = 1
        #         redundantVal=redundantVal-1
        #     if i-radius>=0 and poolingList[i - radius] == 0 and redundantVal>0:
        #         poolingList[i - radius] = 1
        #         redundantVal=redundantVal-1
        #     radius=radius+1
        # poolingList[i] = 1

        # if poolingList[i] >= convWidth * 0.05:
        #     poolingList[i] = 1
        # else:
        #     poolingList[i] = 0

        if poolingList[i] == 2:
            poolingList[i] = 1
            if i + 1 < len(poolingList) and poolingList[i + 1] == 0:
                poolingList[i + 1] = 1
            elif i - 1 >= 0 and poolingList[i - 1] == 0:
                poolingList[i - 1] = 1
        elif poolingList[i] > 2:
            poolingList[i] = 1
            if i + 1 < len(poolingList) and poolingList[i + 1] == 0:
                poolingList[i + 1] = 1
            if i - 1 >= 0 and poolingList[i - 1] == 0:
                poolingList[i - 1] = 1
    flattedList = poolingList
    # print "flattedList: "
    # print numpy.array(flattedList).astype('int16')
    # print flattedList

    out = timeEncoding + flattedList
    # print out
    # print ("finalEncoding length after conv: %s" % len(out))
    return out


def convertWindowsForDataReduction(windows, dataSet, combinedNum):
    timestamp = list(dataSet.data["timestamp"])
    for i in range(0, len(windows)):
        index1 = timestamp.index(windows[i][0])
        leftNum1 = (index1 + 1) % combinedNum
        if leftNum1 != 0:
            if index1 - leftNum1 < 0:
                newTimestamp = timestamp[combinedNum - 1]
            else:
                newTimestamp = timestamp[index1 - leftNum1]
            datetime1 = newTimestamp.strftime('%Y-%m-%d %H:%M:%S')
            datetime1 = datetime.datetime.strptime(datetime1, '%Y-%m-%d %H:%M:%S')
            windows[i][0] = datetime1
        index2 = timestamp.index(windows[i][1])
        leftNum2 = (index2 + 1) % combinedNum
        if leftNum2 != 0:
            if index2 + combinedNum - leftNum2 > dataSet.data.shape[0] - 1:  # 改变的索引超过最大长度
                if dataSet.data.shape[0] % combinedNum == 0:  # 数据集长度能够整除合并数，取最后一个时间戳
                    newTimestamp = timestamp[dataSet.data.shape[0] - 1]
                else:  # 数据集长度能够整除合并数，舍去没有合并的记录，往前取
                    newTimestamp = timestamp[index2 - leftNum2]
            else:
                newTimestamp = timestamp[index2 + combinedNum - leftNum2]
            datetime2 = newTimestamp.strftime('%Y-%m-%d %H:%M:%S')
            datetime2 = datetime.datetime.strptime(datetime2, '%Y-%m-%d %H:%M:%S')
            windows[i][1] = datetime2
        # print windows[i]
    return windows


def convertWindowsForDataReduction_enhence(windows, dataSet, combinedNum):
    timestamp = list(dataSet.data["timestamp"])
    for i in range(0, len(windows)):
        index1 = timestamp.index(windows[i][0])
        index2 = timestamp.index(windows[i][1])
        if index1 < combinedNum - 1:
            if index2 >= combinedNum - 1:
                newTimestamp = timestamp[combinedNum - 1]
                datetime1 = newTimestamp.strftime('%Y-%m-%d %H:%M:%S')
                datetime1 = datetime.datetime.strptime(datetime1, '%Y-%m-%d %H:%M:%S')
                windows[i][0] = datetime1
            else:
                windows.pop(i)
                i -= 1
    return windows
