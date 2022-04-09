# BCDRN

## Running NAB dataset
1. Data preparation. The whole NAB dataset is prepared in `data/`.
2. Select the parameters (e.g. size of convolution core) in `src/run.py`.
3. Run `python main.py`.
4. Check the experiment results in `result/`.

## Running test case
1. Put the test data into `test/data/` (Here we give an example of the water temperature of the crystallizer).
2. Run `python main_test.py`.
3. Check  the scoring results of anomaly detection in `test/result/`.