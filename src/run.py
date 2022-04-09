# coding=utf-8

import os

import util
# from runner import Runner
from runner_enhence import Runner

try:
    import simplejson as json
except ImportError:
    import json

def run():
    filepath = os.path.realpath(__file__)
    root = util.get_nth_parent_dir(1, filepath)
    dataDir = os.path.join(root, "data")
    resultsDir = os.path.join(root, "result")
    nullDir = os.path.join(root, "result", "null")
    windowsFile = os.path.join(root, "label", "combined_windows.json")
    profilesFile = os.path.join(root, "config", "profiles.json")
    thresholdsFile = os.path.join(root, "config", "thresholds.json")

    # module choose
    useDataReduction = True
    useOr = True
    useConv = True
    useAverage = False
    combinedNum = 2
    # convWidth = 2
    # poolingWidth = 3

    # conv core size and pooling size setting
    convWidthStart = 2
    convWidthEnd = 6
    poolingWidthStart = 2
    poolingWidthEnd = 10

    for i in range(convWidthStart, convWidthEnd + 1):
        for j in range(poolingWidthStart, poolingWidthEnd + 1):

            runner = Runner(dataDir=dataDir,
                            labelPath=windowsFile,
                            resultsDir=resultsDir,
                            nullDir=nullDir,
                            profilesPath=profilesFile,
                            thresholdPath=thresholdsFile,
                            numCPUs=None,

                            useDataReduction=useDataReduction,
                            useOr=useOr,
                            useConv=useConv,
                            useAverage=useAverage,
                            combinedNum=combinedNum,
                            convWidth=i,
                            poolingWidth=j
                            )

            runner.initialize()

            runner.detect()

            runner.optimize()

            with open(thresholdsFile) as thresholdConfigFile:
                detectorThresholds = json.load(thresholdConfigFile)
            runner.score(detectorThresholds)

            runner.normalize()


if __name__ == "__main__":
    run()
