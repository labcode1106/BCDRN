import json
import os.path

import src.util as util

import pandas

filepath = os.path.realpath(__file__)
root = util.get_nth_parent_dir(1, filepath)
dataDir = os.path.join(root, "data")
resultsDir = os.path.join(root, "result")
nullDir = os.path.join(root, "result", "null")


windowsFile = os.path.join(root, "label", "combined_windows.json")


# valuationFile=""
# useDataReduction=False
# if useDataReduction:
#     valuationFile = os.path.join(root, "valuation", "valuation_dataReduction.csv")
#     relativeDir="dataReduction"
# else:
#     valuationFile = os.path.join(root, "valuation", "valuation_htm.csv")
#     relativeDir="htm"

valuationFile = os.path.join(root, "valuation", "valuation_dataEnhence.csv")
relativeDir="dataEnhence"

null_standard_scores_Path = os.path.join(nullDir, "null_standard_scores.csv")
null_reward_low_FP_rate_scores_Path = os.path.join(nullDir, "null_reward_low_FP_rate_scores.csv")
null_reward_low_FN_rate_scores_Path = os.path.join(nullDir, "null_reward_low_FN_rate_scores.csv")

dataReduction_standard_scores_Path = os.path.join(resultsDir, relativeDir, "numentaTM_standard_scores.csv")
dataReduction_reward_low_FP_rate_scores_Path = os.path.join(resultsDir, relativeDir,
                                                            "numentaTM_reward_low_FP_rate_scores.csv")
dataReduction_reward_low_FN_rate_scores_Path = os.path.join(resultsDir, relativeDir,
                                                            "numentaTM_reward_low_FN_rate_scores.csv")

valuation_dataEnhence_Path = os.path.join(root, "valuation", "valuation_dataEnhence.csv")
valuation_dataReduction_Path = os.path.join(root, "valuation", "valuation_dataReduction.csv")
valuation_htm_Path = os.path.join(root, "valuation", "valuation_htm.csv")

compare_valuation_Path = os.path.join(root, "valuation", "compare_valuation.csv")

def eval():
    with open(null_standard_scores_Path) as fn:
        nullStandardResults = pandas.read_csv(fn)
        nullStandardResultsLength = nullStandardResults.shape[0]
    with open(dataReduction_standard_scores_Path) as fd:
        dataReductionStandardResults = pandas.read_csv(fd)
        dataReductionStandardResultsLength = dataReductionStandardResults.shape[0]
    with open(windowsFile, "rb") as f:
        labelsDict = json.load(f)

    for i in range(0, nullStandardResultsLength):
        null_standard_score = nullStandardResults["Score"].iloc[i]
        null_standard_dataset = nullStandardResults["File"].iloc[i]
        dt = [null_standard_dataset]
        tpCount = 0
        perfect = 116
        singleScore = 0
        if i == nullStandardResultsLength - 1:
            singleScore = 100 * (dataReductionStandardResults["Score"].iloc[-1] - null_standard_score) / (
                        perfect - null_standard_score)
            dt = ["Totals"]
        for key, value in labelsDict.items():
            if key == null_standard_dataset:
                tpCount = len(value)
                perfect = tpCount * 1
                break
        for j in range(0, dataReductionStandardResultsLength - 1):
            dataReduction_standard_score = dataReductionStandardResults["Score"].iloc[j]
            dataReduction_standard_dataset = dataReductionStandardResults["File"].iloc[j]
            if perfect == 0:
                singleScore = 100
                break
            if null_standard_dataset == dataReduction_standard_dataset:
                singleScore = 100 * (dataReduction_standard_score - null_standard_score) / (
                            perfect - null_standard_score)
                break

        print null_standard_dataset, singleScore

        dt = dt + [singleScore]
        df = pandas.read_csv(valuationFile)
        dfLength = df.shape[0]
        df.loc[dfLength + 1] = dt
        df.to_csv(valuationFile, index=None)

def compare():
    with open(valuation_htm_Path) as fn:
        valuationHtmData = pandas.read_csv(fn)
        valuationHtmDataLength = valuationHtmData.shape[0]
    with open(valuation_dataEnhence_Path) as fn:
        valuationDataReductionData = pandas.read_csv(fn)
        valuationDataReductionDataLength = valuationDataReductionData.shape[0]

    # df = pandas.read_csv(compare_valuation_Path)
    # dfLength = df.shape[0]
    # if df.shape[0]>0:
    #     df.drop([i for i in range(0,dfLength)])

    for i in range(0, valuationHtmDataLength):
        htmDatasetName = valuationHtmData["dataset"].iloc[i]
        htmDatasetStandardScore = valuationHtmData["standard_score"].iloc[i]
        if i<valuationHtmDataLength-1:
            datasetType=(htmDatasetName.split("/"))[0]
            dataset=(htmDatasetName.split("/"))[1]
        else:
            datasetType = ""
            dataset = htmDatasetName
        dt = [datasetType,dataset,htmDatasetStandardScore]
        drScore=None
        increaseRate=None
        for j in range(0, valuationDataReductionDataLength):
            dataReductionDatasetName = valuationDataReductionData["dataset"].iloc[j]
            dataReductionDatasetStandardScore = valuationDataReductionData["standard_score"].iloc[j]

            if htmDatasetName==dataReductionDatasetName:
                drScore=dataReductionDatasetStandardScore
                increaseRate=dataReductionDatasetStandardScore-htmDatasetStandardScore
                print increaseRate
                break

        dt = dt + [drScore, increaseRate]
        df = pandas.read_csv(compare_valuation_Path)
        dfLength = df.shape[0]
        df.loc[dfLength + 1] = dt
        df.to_csv(compare_valuation_Path, index=None)

if __name__ == "__main__":
    # eval()
    compare()
