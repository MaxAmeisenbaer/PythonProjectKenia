# LSTM Neural Network for Time Series Prediction for Real Weather Data from Kenya
## Introduction
The data was gathered over the course of 4 years by multiple weather stations in Kenya as a research project by the Justus Liebig Universität Gießen.
It contains many different weather parameters. You can find information about them in 'Pars.csv'.

The idea is to use LSTM neural networks to predict weather parameters that are traditionally costly to measure and therefore not available in many places.

## Installation

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

## Data
In order to run 'data_preprocessing.py', you need to add a folder called "Data" in the root directory with the corresponding Data.


As of now, preprocessing was only tested on data from the following 3 main stations:
- SHA
- OUT
- NF

TTP only contains the features dish, doc, nit and prec, so it was not yet used for further analysis.


## General ToDo
- Preprocess Data from other stations and make sure the number of time entries is the same for all stations and if not check why and fix it
- Select/find relevant features for the prediction and train models using these features