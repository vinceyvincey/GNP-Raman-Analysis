# (1) Peak Fitting Algorithm

1. Process the raw spectral data using the peak fitting algorithm, with the constraints detailed in the [YAML](constraints/gnp_constraints.yaml) file.

```bash
bash runner.sh
```

2. Once the process is complete concatenate the processed files together. Replace '*example*' in the bash script with desired file name throughout these guides.

```bash
bash concatenate.sh
```

**Note** that this step is time consuming, as there are thousands of spectra to fit. The concatenated processed pickled files are provided in the data store should you wish to avoid it.

* *fluorination_different_gnps.pkl* contains the processed data for the fluorinated GNPs from different manufacturers.

* *puregraph_size_comparison.pkl* contains the processed data for the different sized puregraph GNPs.

3. Copy the concatenated pickled file into the [Data Cleaning](../2_data_cleaning) directory.

```bash
cp data/processed/example.pkl ../2_data_cleaning/data/input
```

4. Only the spectral data is needed for the computer vision work. This can be extracted from the raw .wdf files. Replace '*example_cnn*' with the desired filename at the bottom of the [python script](prepare_data_for_cnn.py).

```  bash
python prepare_data_for_cnn.py

cp data/processed/example_cnn.csv ../4_machine_learning/data/input
```