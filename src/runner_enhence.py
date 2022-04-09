# coding=utf-8

import multiprocessing
import os
import pandas

from src.sweeper import Sweeper

try:
    import simplejson as json
except ImportError:
    import json

from corpus import Corpus
from labeler import CorpusLabel
from optimizer import optimizeThreshold
from scorer import scoreCorpus
import nabUtil
import util
from detector import Detector
from detector_enhence import Detector_enhence


class Runner(object):
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
                 poolingWidth,

                 dataDir,
                 resultsDir,
                 nullDir,
                 labelPath,
                 profilesPath,
                 thresholdPath,
                 numCPUs=None,
                 ):
        """
        @param dataDir        (string)  Directory where all the raw datasets exist.

        @param resultsDir     (string)  Directory where the detector anomaly scores
                                        will be scored.

        @param labelPath      (string)  Path where the labels of the datasets
                                        exist.

        @param profilesPath   (string)  Path to JSON file containing application
                                        profiles and associated cost matrices.

        @param thresholdPath  (string)  Path to thresholds dictionary containing the
                                        best thresholds (and their corresponding
                                        score) for a combination of detector and
                                        user profile.

        @probationaryPercent  (float)   Percent of each dataset which will be
                                        ignored during the scoring process.

        @param numCPUs        (int)     Number of CPUs to be used for calls to
                                        multiprocessing.pool.map
        """
        self.dataDir = dataDir
        self.resultsDir = resultsDir
        self.nullDir = nullDir
        self.labelPath = labelPath
        self.profilesPath = profilesPath
        self.thresholdPath = thresholdPath
        self.pool = multiprocessing.Pool(numCPUs)

        self.probationaryPercent = 0.15
        self.windowSize = 0.10

        self.corpus = None
        self.corpusLabel = None
        self.profiles = None

        self.detectorName = "numentaTM"

        # self.useConv = True
        # self.useAverage = False
        # self.combinedNum = 2
        # self.convWidth = 2
        # self.poolingWidth = 5


        self.useDataReduction = useDataReduction
        self.useOr = useOr
        self.useConv = useConv
        self.useAverage = useAverage
        self.combinedNum = combinedNum
        self.convWidth = convWidth
        self.poolingWidth = poolingWidth


    def initialize(self):
        """Initialize all the relevant objects for the run."""
        self.corpus = Corpus(self.dataDir)
        self.corpusLabel = CorpusLabel(path=self.labelPath, corpus=self.corpus)

        with open(self.profilesPath) as p:
            self.profiles = json.load(p)

        self.methodDir = "dataEnhence"

    def detect(self):
        print("\nRunning detect step")
        count = 0
        for relativePath, dataSet in self.corpus.dataFiles.iteritems():
            if relativePath not in self.corpusLabel.labels:
                continue
            labels = self.corpusLabel.labels[relativePath]
            if self.combinedNum>1:
                labels = util.convertLabelsBasedOnDataReduction_enhence(labels, self.combinedNum)
            relativeDir, fileName = os.path.split(relativePath)
            resultFileName = self.detectorName + "_" + fileName
            outputPath = os.path.join(self.resultsDir, self.methodDir, relativeDir, resultFileName)
            nabUtil.createPath(outputPath)

            detector = Detector_enhence(self.useConv, self.useAverage, self.combinedNum, self.convWidth, self.poolingWidth)
            detector.anomaly_detect(dataSet, labels, outputPath, self.probationaryPercent)

            print "%s: Detection done for %s" % (count, os.path.join(relativeDir, resultFileName))
            count = count + 1

    def optimize(self):
        """Optimize the threshold for each combination of detector and profile.

        @param detectorNames  (list)  List of detector names.

        @return thresholds    (dict)  Dictionary of dictionaries with detector names
                                      then profile names as keys followed by another
                                      dictionary containing the score and the
                                      threshold used to obtained that score.
        """
        print("\n\nRunning optimize step")

        scoreFlag = False
        thresholds = {}

        resultsDetectorDir = os.path.join(self.resultsDir, self.methodDir)
        resultsCorpus = Corpus(resultsDetectorDir)

        thresholds[self.detectorName] = {}

        for profileName, profile in self.profiles.items():
            allAnomalyRows = []
            costMatrix = profile["CostMatrix"]
            # windows = corpusLabel.windows[datasetName]
            # if useDataReduction and useOr:
            #     windows = util.convertWindowsForDataReduction(windows, dataSet, combinedNum)
            # labels = corpusLabel.labels[datasetName]
            # if useDataReduction and useOr:
            #     labels = util.convertLabelsBasedOnDataReduction(labels, combinedNum)
            # timestamps = labels['timestamp']
            # resultDataSet = resultsCorpus.dataFiles[resultName]
            # anomalyScores = resultDataSet.data["anomaly_score"]

            sweeper = Sweeper(
                probationPercent=self.probationaryPercent,
                costMatrix=costMatrix
            )

            for relativePath, dataSet in resultsCorpus.dataFiles.items():
                if "_scores.csv" in relativePath:
                    continue

                # relativePath: raw dataset file,
                # e.g. 'artificialNoAnomaly/art_noisy.csv'
                relativePath = nabUtil.convertResultsPathToDataPath(
                    os.path.join(self.detectorName, relativePath))

                windows = self.corpusLabel.windows[relativePath]
                if self.combinedNum>1:
                    windows = util.convertWindowsForDataReduction(windows, self.corpus.dataFiles[relativePath],
                                                                  self.combinedNum)
                labels = self.corpusLabel.labels[relativePath]
                if self.combinedNum>1:
                    labels = util.convertLabelsBasedOnDataReduction_enhence(labels, self.combinedNum)
                # windows = corpusLabel.windows[relativePath]
                # labels = corpusLabel.labels[relativePath]
                timestamps = labels['timestamp']
                anomalyScores = dataSet.data["anomaly_score"]

                curAnomalyRows = sweeper.calcSweepScore(
                    timestamps,
                    anomalyScores,
                    windows,
                    relativePath
                )
                allAnomalyRows.extend(curAnomalyRows)

            # curAnomalyRows = sweeper.calcSweepScore(
            #     timestamps,
            #     anomalyScores,
            #     windows,
            #     datasetName
            # )
            # allAnomalyRows.extend(curAnomalyRows)

            # Get scores by threshold for the entire corpus
            scoresByThreshold = sweeper.calcScoreByThreshold(allAnomalyRows)
            scoresByThreshold = sorted(
                scoresByThreshold, key=lambda x: x.score, reverse=True)
            bestParams = scoresByThreshold[0]

            print(("Optimizer found a max score of {} with anomaly threshold {}.".format(
                bestParams.score, bestParams.threshold
            )))

            thresholds[self.detectorName][profileName] = {
                "threshold": bestParams.threshold,
                "score": bestParams.score
            }

        nabUtil.updateThresholds(thresholds, self.thresholdPath)

        return thresholds

    def score(self, detectorThresholds):
        print("\n\nRunning score step")
        scoreFlag = True

        resultsDetectorDir = os.path.join(self.resultsDir, self.methodDir)
        resultsCorpus = Corpus(resultsDetectorDir)

        self.resultsFiles = []
        for profileName, profile in self.profiles.items():
            threshold = detectorThresholds[self.detectorName][profileName]["threshold"]
            costMatrix = profile["CostMatrix"]
            results = []
            for relativePath, dataSet in resultsCorpus.dataFiles.items():
                if "_scores.csv" in relativePath:
                    continue

                # relativePath: raw dataset file,
                # e.g. 'artificialNoAnomaly/art_noisy.csv'
                relativePath = nabUtil.convertResultsPathToDataPath(
                    os.path.join(self.detectorName, relativePath))

                # outputPath: dataset results file,
                # e.g. 'results/detector/artificialNoAnomaly/detector_art_noisy.csv'
                relativeDir, fileName = os.path.split(relativePath)
                fileName = self.detectorName + "_" + fileName
                outputPath = os.path.join(resultsDetectorDir, relativeDir, fileName)

                # windows = corpusLabel.windows[relativePath]
                # labels = corpusLabel.labels[relativePath]
                windows = self.corpusLabel.windows[relativePath]
                if self.combinedNum>1:
                    windows = util.convertWindowsForDataReduction_enhence(windows, self.corpus.dataFiles[relativePath],
                                                                  self.combinedNum)
                labels = self.corpusLabel.labels[relativePath]
                if self.combinedNum>1:
                    labels = util.convertLabelsBasedOnDataReduction_enhence(labels, self.combinedNum)
                timestamps = labels['timestamp']

                anomalyScores = dataSet.data["anomaly_score"]

                scorer = Sweeper(
                    probationPercent=self.probationaryPercent,
                    costMatrix=costMatrix
                )

                (scores, bestRow) = scorer.scoreDataSet(
                    timestamps,
                    anomalyScores,
                    windows,
                    relativePath,
                    threshold,
                )
                if scoreFlag:
                    # Append scoring function values to the respective results file
                    dfCSV = pandas.read_csv(outputPath, header=0, parse_dates=[0])
                    dfCSV["S(t)_%s" % profileName] = scores
                    dfCSV.to_csv(outputPath, index=False)

                # print("S(t)_%s: %s" % (profileName, scores))
                # print("bestRow: %s" % str(bestRow))

                sweeperResult = (self.detectorName, profileName, relativePath, threshold, bestRow.score,
                                  bestRow.tp, bestRow.tn, bestRow.fp, bestRow.fn, bestRow.total)
                results.append(sweeperResult)

            totals = [None] * 3 + [0] * 6
            for row in results:
                for i in range(6):
                    totals[i + 3] += row[i + 4]

            results.append(["Totals"] + totals)

            resultsDF = pandas.DataFrame(data=results,
                                         columns=("Detector", "Profile", "File",
                                                  "Threshold", "Score", "TP", "TN",
                                                  "FP", "FN", "Total_Count"))

            scorePath = os.path.join(resultsDetectorDir, "%s_%s_scores.csv" % (self.detectorName, profileName))
            resultsDF.to_csv(scorePath, index=False)
            print("%s detector benchmark scores written to %s" % (self.detectorName, scorePath))
            self.resultsFiles.append(scorePath)

    def normalize(self):
        print("\n\nRunning normalization step")

        # Get baseline scores for each application profile.
        if not os.path.isdir(self.nullDir):
            raise IOError("No results directory for null detector. You must "
                          "run the null detector before normalizing scores.")

        # resultsFiles = []
        baselines = {}
        for profileName, _ in self.profiles.items():
            fileName = os.path.join(self.nullDir,
                                    "null_" + profileName + "_scores.csv")
            with open(fileName) as f:
                results = pandas.read_csv(f)
                baselines[profileName] = results["Score"].iloc[-1]

        # Get total number of TPs
        with open(self.labelPath, "rb") as f:
            labelsDict = json.load(f)
        tpCount = 0
        for labels in list(labelsDict.values()):
            tpCount += len(labels)

        # Normalize the score from each results file.
        # finalResults = {}
        finalScores=[]
        for resultsFile in self.resultsFiles:
            profileName = [k for k in list(baselines.keys()) if k in resultsFile][0]
            base = baselines[profileName]

            with open(resultsFile) as f:
                results = pandas.read_csv(f)

                # Calculate score:
                perfect = tpCount * self.profiles[profileName]["CostMatrix"]["tpWeight"]
                score = 100 * (results["Score"].iloc[-1] - base) / (perfect - base)
                finalScores.append(score)

                # Add to results dict:
                resultsInfo = resultsFile.split(os.path.sep)[-1].split('.')[0]
                detector = resultsInfo.split('_')[0]
                profile = resultsInfo.replace(detector + "_", "").replace("_scores", "")
                # if detector not in finalResults:
                #     finalResults[detector] = {}
                # finalResults[detector][profile] = score

            print(("Final score for \'%s\' detector on \'%s\' profile = %.5f"
                   % (detector, profile, score)))

        convWidthResult = ""
        poolingWidthResult = ""
        if self.useConv:
            convWidthResult = self.convWidth
            poolingWidthResult = self.poolingWidth
        if self.useAverage:
            convWidthResult = self.convWidth

        dt = ["combine", self.combinedNum, convWidthResult, poolingWidthResult, str(self.useAverage)] + finalScores
        finalScorePath = os.path.join(self.resultsDir, "final_score.csv")
        df = pandas.read_csv(finalScorePath)
        dfLength = df.shape[0]
        df.loc[dfLength + 1] = dt
        df.to_csv(finalScorePath, index=None)
