# coding=utf-8

import csv
import datetime
import math

import pandas
import numpy

from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.backtracking_tm_shim import TMCPPShim as TM
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.algorithms import anomaly_likelihood
from nupic.algorithms import anomaly

import src.util

try:
    import simplejson as json
except ImportError:
    import json

SPATIAL_TOLERANCE = 0.05
minResolution = 0.001
numBuckets = 130.0


class Detector(object):

    def __init__(self, useConv, useAverage, combinedNum, convWidth, poolingWidth):
        self.combinedNum=combinedNum
        self.useConv=useConv
        self.useAverage=useAverage
        self.convWidth=convWidth
        self.poolingWidth = poolingWidth

    def calResolutionForEncoder(self, minVal, maxVal, minResolution, numBuckets):
        resolution = max(minResolution, (maxVal - minVal) / numBuckets)
        print ("resolution: %s" % resolution)
        return resolution

    def anomaly_detect(self, dataSet, outputPath, probationaryPercent):
        global numBuckets
        inputMin = dataSet.data["value"].min()
        inputMax = dataSet.data["value"].max()
        rangePadding = abs(inputMax - inputMin) * 0.2
        minVal = inputMin - rangePadding
        maxVal = inputMax + rangePadding

        timeOfDayEncoder = DateEncoder(timeOfDay=(21, 9.49))
        scalarEncoder = RandomDistributedScalarEncoder(
            self.calResolutionForEncoder(minVal, maxVal, minResolution, numBuckets*2),
            w=21,
            n=400)
        timeOfDayEncoderWidth = timeOfDayEncoder.getWidth()
        scalarEncoderWidth = scalarEncoder.getWidth()

        """  # Creating the SP   """
        encodingWidth = timeOfDayEncoderWidth + scalarEncoderWidth
        if self.useConv:
            encodingWidth = timeOfDayEncoderWidth + ((scalarEncoderWidth - self.convWidth + 1) + self.poolingWidth - 1) / self.poolingWidth

        print("encodingWidth: %s" % encodingWidth)

        sp = SpatialPooler(
            inputDimensions=(encodingWidth),
            columnDimensions=(2048),
            potentialRadius=2048,
            potentialPct=0.8,
            globalInhibition=True,
            localAreaDensity=-1.0,
            numActiveColumnsPerInhArea=40.0,
            synPermInactiveDec=0.0005,
            synPermActiveInc=0.003,
            synPermConnected=0.1,
            boostStrength=0.0,
            seed=1956,
            wrapAround=False
        )

        """  Creating the TM   """
        tm = TM(numberOfCols=2048,
                cellsPerColumn=32,
                activationThreshold=20,
                minThreshold=13,
                initialPerm=0.24,
                connectedPerm=0.50,
                permanenceInc=0.04,
                permanenceDec=0.008,
                predictedSegmentDecrement=0.001,
                newSynapseCount=31,
                globalDecay=0.0,
                seed=1960,
                verbosity=0,
                # pamLength=3,
                maxAge=0,
                maxSegmentsPerCell=128,
                maxSynapsesPerSegment=128,
                outputType='normal',
                )

        dataSetLength = dataSet.data.shape[0]

        probationaryPeriod = src.util.getProbationPeriod(probationaryPercent, dataSetLength)
        numentaLearningPeriod = int(math.floor(probationaryPeriod / 2.0))
        anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
            learningPeriod=numentaLearningPeriod,
            estimationSamples=probationaryPeriod - numentaLearningPeriod,
            reestimationPeriod=100)

        headers = ["timestamp", "value", "anomaly_score", "raw_score"]
        outputRows = []
        curMinVal = None
        curMaxVal = None

        print('start detection...')

        encodingList = []
        for i, row in dataSet.data.iterrows():
            if i%5000==0 and i!=0:
                print 'finish detecting num: ',i
            inputData = row.to_dict()
            timestamp = inputData['timestamp']
            value = inputData['value']
            # print('\ninput %s: %s, %s' % (i, timestamp, value))

            """  Encoding Data   """
            timeOfDayBits = numpy.zeros(timeOfDayEncoder.getWidth())
            consumptionBits = numpy.zeros(scalarEncoder.getWidth())

            timeOfDayEncoder.encodeIntoArray(timestamp, timeOfDayBits)
            scalarEncoder.encodeIntoArray(value, consumptionBits)

            encoding = numpy.concatenate(
                [timeOfDayBits, consumptionBits]
            )

            if i < self.combinedNum-1:
                encodingList.append(encoding)
                continue
            encodingList.append(encoding)
            combinedEncoding = encodingList[0]
            for i in range(1, len(encodingList)):
                combinedEncoding = src.util.calOrForTwoList(encodingList[i], combinedEncoding)
            encodingList.pop(0)

            if self.useConv:
                combinedEncoding = src.util.calConvoluteEncoding(combinedEncoding, self.convWidth, self.poolingWidth, timeOfDayEncoderWidth)
            if self.useAverage:
                combinedEncoding = src.util.calAverageEncoding(combinedEncoding, self.convWidth, timeOfDayEncoderWidth)
            else:
                combinedEncoding = src.util.convertEncodingTo0And1(combinedEncoding, timeOfDayEncoderWidth)

            """  SP and TM   """
            column = numpy.zeros(2048, dtype=numpy.int32)
            sp.compute(numpy.array(combinedEncoding), True, column)
            prdictiveColumnsSdr = tm.topDownCompute().copy()
            prdictiveColumns = prdictiveColumnsSdr.nonzero()[0]
            tm.compute(column, True, True)
            activateColumns = numpy.nonzero(column)[0]
            activateColumns = activateColumns.astype(numpy.int32)
            raw_score = anomaly.computeRawAnomalyScore(activateColumns, prdictiveColumns)
            # print ("raw_score: %s" % raw_score)

            spatialAnomaly = False
            # value = consumption
            if curMinVal != curMaxVal:
                tolerance = (curMaxVal - curMinVal) * SPATIAL_TOLERANCE
                maxExpected = curMaxVal + tolerance
                minExpected = curMinVal - tolerance
                if value > maxExpected or value < minExpected:
                    spatialAnomaly = True
            if curMaxVal is None or value > curMaxVal:
                curMaxVal = value
            if curMinVal is None or value < curMinVal:
                curMinVal = value

            """  calculate LogLikelihood(anomaly_score)  """
            anomalyScore = anomalyLikelihood.anomalyProbability(value, raw_score, timestamp)
            # print("anomalyScore: %s" % anomalyScore)
            logScore = anomalyLikelihood.computeLogLikelihood(anomalyScore)

            if spatialAnomaly:
                logScore = 1.0
            # print("logScore: %s" % logScore)

            """  store result  """
            calResult = (logScore, raw_score)
            outputRow = list(row) + list(calResult)
            outputRows.append(outputRow)


        results = pandas.DataFrame(outputRows, columns=headers)
        results.to_csv(outputPath, index=False)
