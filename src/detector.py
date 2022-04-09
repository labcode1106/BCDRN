# coding=utf-8

import csv
import datetime
import math

import pandas
import numpy
import os
import yaml

from nupic.algorithms.sdr_classifier_factory import SDRClassifierFactory
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory_shim import TemporalMemoryShim
from nupic.algorithms.backtracking_tm_shim import TMCPPShim as TM
from nupic.algorithms.sdr_classifier import SDRClassifier
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.algorithms import anomaly_likelihood
from nupic.algorithms import anomaly

import util
import nabUtil
from corpus import Corpus
from labeler import CorpusLabel
from scorer import scoreCorpus
from sweeper import Sweeper
from optimizer import optimizeThreshold

try:
    import simplejson as json
except ImportError:
    import json


SPATIAL_TOLERANCE = 0.05
minResolution = 0.001
numBuckets = 130.0

# def calAnomalyScore(P, A):
#     ALength = len(A)
#     AAndPRepetitionCount = util.calRepetitionCountInList(P, A)
#     # print(ALength)
#     # print(AAndPRepetitionCount)
#     anomalyScore = float(ALength - AAndPRepetitionCount) / float(ALength)
#     return anomalyScore


class Detector(object):
    """
    Class to run an endpoint (detect, optimize, or score) on the NAB
    benchmark using the specified set of profiles, thresholds, and/or detectors.
    """

    def __init__(self,
                 useDataReduction,
                 useOr,
                 useConv,
                 useAverage,
                 combinedNum,
                 convWidth,
                 poolingWidth):

        self.useDataReduction = useDataReduction
        self.useOr = useOr
        self.useConv = useConv
        self.useAverage = useAverage
        self.combinedNum = combinedNum  # combined records num
        self.convWidth = convWidth  # size of conv core
        self.poolingWidth = poolingWidth    # size of pooling core


    def calResolutionForEncoder(self, minVal, maxVal, minResolution, numBuckets):
        resolution = max(minResolution, (maxVal - minVal) / numBuckets)
        print ("resolution: %s" % resolution)
        return resolution


    def anomaly_detect(self, dataSet, labels, outputPath, probationaryPercent):
        global numBuckets
        inputMin = dataSet.data["value"].min()
        inputMax = dataSet.data["value"].max()
        rangePadding = abs(inputMax - inputMin) * 0.2
        minVal = inputMin - rangePadding
        maxVal = inputMax + rangePadding

        timeOfDayEncoder = DateEncoder(timeOfDay=(21, 9.49))
        curNumBuckets = numBuckets
        if self.useDataReduction and self.useOr:
            curNumBuckets = numBuckets * self.combinedNum
        scalarEncoder = RandomDistributedScalarEncoder(self.calResolutionForEncoder(minVal, maxVal, minResolution, curNumBuckets*2),
                                                       w=21,
                                                       n=400)
        timeOfDayEncoderWidth = timeOfDayEncoder.getWidth()
        scalarEncoderWidth = scalarEncoder.getWidth()

        # print("timeOfDayEncoder width: %s" % timeOfDayEncoderWidth)
        # print("scalarEncoder width: %s" % scalarEncoderWidth)

        """  # Creating the SP   """
        encodingWidth = timeOfDayEncoderWidth + scalarEncoderWidth
        if self.useDataReduction and self.useConv:
            encodingWidth = timeOfDayEncoderWidth + ((scalarEncoderWidth - self.convWidth + 1) + self.poolingWidth - 1) / self.poolingWidth
        print("encodingWidth: %s" % encodingWidth)

        # global transmissionAmount
        # transmissionAmount = encodingWidth
        # print transmissionAmount

        sp = SpatialPooler(
            # How large the input encoding will be.
            inputDimensions=(encodingWidth),
            # How many mini-columns will be in the Spatial Pooler.
            columnDimensions=(2048),
            potentialRadius=2048,
            # What percent of the columns's receptive field is available for potential
            # synapses?
            potentialPct=0.8,
            # This means that the input space has no topology.
            globalInhibition=True,
            localAreaDensity=-1.0,
            # Roughly 2%, giving that there is only one inhibition area because we have
            # turned on globalInhibition (40 / 2048 = 0.0195)
            numActiveColumnsPerInhArea=40.0,
            # How quickly synapses grow and degrade.
            synPermInactiveDec=0.0005,
            synPermActiveInc=0.003,
            synPermConnected=0.1,
            # boostStrength controls the strength of boosting. Boosting encourages
            # efficient usage of SP columns.
            boostStrength=0.0,
            # Random number generator seed.
            seed=1956,
            # Determines if inputs at the beginning and end of an input dimension should
            # be considered neighbors when mapping columns to inputs.
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

        # tm = TemporalMemoryShim(columnDimensions=(2048,),
        #                         cellsPerColumn=32,
        #                         activationThreshold=20,
        #                         initialPermanence=0.24,
        #                         connectedPermanence=0.50,
        #                         minThreshold=13,
        #                         maxNewSynapseCount=20,
        #                         permanenceIncrement=0.04,
        #                         permanenceDecrement=0.008,
        #                         seed=1960
        #                         )

        # finLength = 0
        # with open(_INPUT_FILE_PATH) as fin:
        #     finLength = len(fin.readlines())

        dataSetLength = dataSet.data.shape[0]

        # probationaryPeriod = util.getProbationPeriod(probationaryPercent, finLength)
        probationaryPeriod = util.getProbationPeriod(probationaryPercent, dataSetLength)
        numentaLearningPeriod = int(math.floor(probationaryPeriod / 2.0))
        anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
            learningPeriod=numentaLearningPeriod,
            estimationSamples=probationaryPeriod - numentaLearningPeriod,
            reestimationPeriod=100)
        # print("estimationSamples: %s" % (probationaryPeriod - numentaLearningPeriod))
        # print("probationaryPeriod, numentaLearningPeriod: %s, %s" % (
        #     probationaryPeriod, numentaLearningPeriod))

        prdictiveColumns = numpy.zeros(2048)
        headers = ["timestamp", "value", "anomaly_score", "raw_score"]
        outputRows = []
        combinedencodings = []
        curCombinedNum = 0
        curMinVal = None
        curMaxVal = None
        # SDRclassifier = SDRClassifier(steps=[1], alpha=0.035828933612158, actValueAlpha=0.1, verbosity=0)

        for i, row in dataSet.data.iterrows():
            inputData = row.to_dict()
            timestamp=inputData['timestamp']
            value=inputData['value']
            # print('\ninput %s: %s, %s' % (i, timestamp, value))

            """  Encoding Data   """
            timeOfDayBits = numpy.zeros(timeOfDayEncoder.getWidth())
            consumptionBits = numpy.zeros(scalarEncoder.getWidth())

            timeOfDayEncoder.encodeIntoArray(timestamp, timeOfDayBits)
            scalarEncoder.encodeIntoArray(value, consumptionBits)

            encoding = numpy.concatenate(
                [timeOfDayBits, consumptionBits]
            )

            """ data reduction """
            finalEncoding = encoding
            if self.useDataReduction:
                if self.useOr:
                    if curCombinedNum == 0:  # start combining
                        combinedencodings = encoding
                        curCombinedNum += 1
                        continue
                    elif curCombinedNum < self.combinedNum and i < dataSetLength:  # less than target combining num, conduct 'or'
                        # changedTimeEncoding = util.calAndForTwoList(combinedencodings[0:timeOfDayEncoderWidth],
                        #                                             encoding[0:timeOfDayEncoderWidth])

                        # changedTimeEncoding=encoding.tolist()
                        # changedTimeEncoding = changedTimeEncoding[0:timeOfDayEncoderWidth]
                        # changedScalerEncodings = util.calOrForTwoList(combinedencodings[timeOfDayEncoderWidth::],
                        #                                               encoding[timeOfDayEncoderWidth::])
                        # combinedencodings = changedTimeEncoding + changedScalerEncodings

                        combinedencodings = util.calOrForTwoList(combinedencodings, encoding)
                        curCombinedNum += 1
                        # continue
                    if curCombinedNum == self.combinedNum:  # enough records, initialize the combined encoding
                        finalEncoding = combinedencodings
                        curCombinedNum = 0
                        combinedencodings = []
                    else:
                        continue
                # print numpy.array(finalEncoding).astype('int16')
                if self.useConv:
                    finalEncoding = util.calConvoluteEncoding(finalEncoding, self.convWidth, self.poolingWidth, timeOfDayEncoderWidth)
                if self.useAverage:
                    finalEncoding = util.calAverageEncoding(finalEncoding, self.convWidth, timeOfDayEncoderWidth)
                else:
                    finalEncoding = util.convertEncodingTo0And1(finalEncoding, timeOfDayEncoderWidth)

            """  SP and TM   """
            column = numpy.zeros(2048, dtype=numpy.int32)
            sp.compute(numpy.array(finalEncoding), True, column)
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
        results["label"] = labels["label"]
        results.to_csv(outputPath, index=False)

        # return transmissionAmount

        # with open(_INPUT_FILE_PATH) as fin:
        #     reader = csv.reader(fin)
        #     for count, record in enumerate(reader):
        #         print('\ninput %s: %s, %s' % (count, record[0], record[1]))
        #         if count == 0:
        #             continue
        #         dateString = datetime.datetime.strptime(record[0][0:-3], '%Y-%m-%d %H:%M')
        #         consumption = float(record[1])
        #
        #         """  Encoding Data   """
        #         timeOfDayBits = numpy.zeros(timeOfDayEncoder.getWidth())
        #         consumptionBits = numpy.zeros(scalarEncoder.getWidth())
        #
        #         timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
        #         scalarEncoder.encodeIntoArray(consumption, consumptionBits)
        #
        #         encoding = numpy.concatenate(
        #             [timeOfDayBits, consumptionBits]
        #         )
        #
        #         # print("Encoding Data encoding: ")
        #         # print(encoding.astype('int16'))
        #         # print (timeOfDayEncoder)
        #         # print (scalarEncoder)
        #
        #         """ data reduction """
        #         if useDataReduction:
        #             finalEncoding = encoding
        #             if useOr:
        #                 if curCombinedNum == 0:
        #                     combinedencodings = encoding
        #                     curCombinedNum += 1
        #                     continue
        #                 elif curCombinedNum < combinedNum and count < dataSetLength:
        #                     combinedencodings = util.calOrForTwoList(combinedencodings, encoding)
        #                     curCombinedNum += 1
        #                     # continue
        #                 if curCombinedNum == combinedNum:
        #                     finalEncoding = combinedencodings
        #                     curCombinedNum = 0
        #                     combinedencodings = []
        #                 else:
        #                     continue
        #             # print numpy.array(finalEncoding).astype('int16')
        #             if useConv:
        #                 finalEncoding = util.calConvoluteEncoding(finalEncoding, convWidth, timeOfDayEncoderWidth)
        #             if useAverage:
        #                 finalEncoding = util.calAverageEncoding(finalEncoding, convWidth, timeOfDayEncoderWidth)
        #             else:
        #                 finalEncoding = util.convertEncodingTo0And1(finalEncoding, timeOfDayEncoderWidth)
        #
        #         """  SP and TM   """
        #         column = numpy.zeros(2048, dtype=numpy.int32)
        #         sp.compute(numpy.array(finalEncoding), True, column)
        #         prdictiveColumnsSdr = tm.topDownCompute().copy()
        #         prdictiveColumns = prdictiveColumnsSdr.nonzero()[0]
        #         tm.compute(column, True, True)
        #         activateColumns = numpy.nonzero(column)[0]
        #         activateColumns = activateColumns.astype(numpy.int32)
        #         raw_score = anomaly.computeRawAnomalyScore(activateColumns, prdictiveColumns)
        #         print ("raw_score: %s" % raw_score)
        #
        #         spatialAnomaly = False
        #         value = consumption
        #         if curMinVal != curMaxVal:
        #             tolerance = (curMaxVal - curMinVal) * SPATIAL_TOLERANCE
        #             maxExpected = curMaxVal + tolerance
        #             minExpected = curMinVal - tolerance
        #             if value > maxExpected or value < minExpected:
        #                 spatialAnomaly = True
        #         if curMaxVal is None or value > curMaxVal:
        #             curMaxVal = value
        #         if curMinVal is None or value < curMinVal:
        #             curMinVal = value
        #
        #         """  calculate LogLikelihood(anomaly_score)  """
        #         anomalyScore = anomalyLikelihood.anomalyProbability(record[1], raw_score, record[0])
        #         # print("anomalyScore: %s" % anomalyScore)
        #         logScore = anomalyLikelihood.computeLogLikelihood(anomalyScore)
        #
        #         if spatialAnomaly:
        #             logScore = 1.0
        #         print("logScore: %s" % logScore)
        #
        #         """  store result  """
        #         calResult = (logScore, raw_score)
        #         outputRow = list(record) + list(calResult)
        #         outputRows.append(outputRow)
        #
        #     results = pandas.DataFrame(outputRows, columns=headers)
        #     if useDataReduction and useOr:
        #         labels = util.convertLabelsBasedOnDataReduction(labels, combinedNum)
        #     results["label"] = labels["label"]
        #     if useDataReduction:
        #         results.to_csv(_RESULT_REDUCTION_FILE_PATH, index=False)
        #     else:
        #         results.to_csv(_RESULT_FILE_PATH, index=False)