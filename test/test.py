# coding=utf-8

import os
import src.util
from src.runner_enhence import Runner

from src.corpus import Corpus
from detect import Detector

try:
    import simplejson as json
except ImportError:
    import json



def run():
    filepath = os.path.realpath(__file__)
    root = src.util.get_nth_parent_dir(1, filepath)
    dataDir = os.path.join(root, "test", "data")
    resultsDir = os.path.join(root, "test", "result")
    outputPath = os.path.join(resultsDir, "crystallizer_temperature.csv")

    corpus = Corpus(dataDir)
    # print('finish Corpus')
    for relativePath, dataSet in corpus.dataFiles.iteritems():
        detector = Detector(True, False, 2, 2, 3)
        detector.anomaly_detect(dataSet, outputPath, 0.15)


if __name__ == "__main__":
    run()
