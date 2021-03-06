# MmWave Classification
Imperial College Masters Individual Project 

This project is a first step towards a CWD system, with the utlimate aim of classifying concealed weapons over a certain range, eliminating the need for portals or handheld devices in security screenings for exmple.

In this project, we attempted to demonstrate the reliability of Deep Learning techniques applied to data from TI's mmwave sensor (IWR1443). We succesfully classified hanheld spoon from a knife, in various conditions.

Note: The collected datasets and models are too large to upload via git, please contact me if you wish to have access.

## Report

The [report](https://github.com/Kheil-Z/MmWave_Classification/blob/main/final_report.pdf) goes further into detail concerning the project.
## Contents

- The [aux directory](https://github.com/Kheil-Z/MmWave_Classification/tree/main/aux) contains auxiliary files with functions necessary for the main code structure. These are used to parse the config file, parse the raw data once streamed, implement Peak Grouping on the objects data class(also voxelize it), and contain the Pytorch Models structures.

- the [toCSV directory](https://github.com/Kheil-Z/MmWave_Classification/tree/main/toCSV) contins all required code to transform parsed data from .txt files to appropriate .csv files for learning using pytorch and the learning notebook. (Note: [toCSV_heatmapROI](https://github.com/Kheil-Z/MmWave_Classification/blob/main/toCSV/toCSV_heatmapROI.py) creates a data structure with only the heatmap ROI, meanwhile [toCSV_heatmapROI_voxel](https://github.com/Kheil-Z/MmWave_Classification/blob/main/toCSV/toCSV_heatmapROI_voxel.py) creates a data structure containing both the heatmap ROI and voxelized objects structure. The latter works best, and the the notebook is configured to work with it.

-All files ending in .pt are trained pytorch models retained, [the one](https://github.com/Kheil-Z/MmWave_Classification/blob/main/3u6vs4fl_11.pt) we discuss in the paper (which we will upload soon), is the one currently in use in the prediction file, you can commemnt it and use any model you wish.

- [The configuration file](https://github.com/Kheil-Z/MmWave_Classification/blob/main/knife3dNoGrouping.cfg)  we used can be changed, this is generated using [TI's tool](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/2.1.0/).

- [The notebook](https://github.com/Kheil-Z/MmWave_Classification/blob/main/Training/MmwaveLearning_Project.ipynb) used to train the models is all ready to run on data issued from the [toCSV_heatmapROI_voxel.py](https://github.com/Kheil-Z/MmWave_Classification/blob/main/toCSV/toCSV_heatmapROI_voxel.py) code after collecting data( or using our data)

- Finally, the [main code structure](https://github.com/Kheil-Z/MmWave_Classification/blob/main/predict.py) is used to either stream data live and save it, or just visualize it, or to make live predictions on the streamed data once the sensor is connected.

## Usage

All codes contain a "main" loop with comments indicating their appropraite usage. Furthemore, inputing "h" will print a small help page when using the [prediction code](https://github.com/Kheil-Z/MmWave_Classification/blob/main/predict.py), this should explain how to use each mode appropriately.
