# EV Battery RUL Predictor
A machine learning project that predicts the remaining useful life (RUL) of electric vehicle batteries using CNN-LSTM neural networks.

## Overview
This project builds a predictive model to estimate battery remaining useful life cycles based on various sensor readings throughout the battery's discharge cycles. The model learns degradation patterns from historical battery data and can forecast when a battery will reach end-of-life (70% of initial capacity).

## Features
- Data loading and preprocessing from MATLAB (.mat) battery datasets
- Multi-level signal normalization (input and target scaling)
- Sliding window dataset creation for temporal sequence learning
- CNN-LSTM hybrid architecture for spatial and temporal feature extraction
- Multi-battery training for improved generalization
- Model evaluation with visualization of actual vs predicted RUL
- Support for GPU acceleration with PyTorch

## Project Structure
- **ev_battery_rlu.ipynb**: Main Jupyter notebook containing the complete pipeline
- **5. Battery Data Set/**: Directory containing battery discharge cycle data
  - B0005.mat, B0006.mat, B0007.mat, B0018.mat: Battery datasets from NASA

## Technical Architecture
- **CNN Component**: Extracts spatial features from voltage, current, and temperature sensor signals
- **LSTM Component**: Captures temporal degradation patterns across multiple discharge cycles
- **Data Processing**: 1D signal resampling to fixed 200-point sequences for consistent model input

## Key Improvements
1. Dynamic key access for handling variable MATLAB file structures
2. Dual-level scaling (input and target normalization) for stable gradient flow
3. Sliding window approach (30 cycles) for temporal context
4. Dropout regularization to prevent overfitting across multiple batteries

## Model Performance
The model successfully captures battery degradation trends with the ability to:
- Predict descending slopes of battery lifecycles
- Generalize to unseen batteries
- Handle real-world sensor noise
- Identify degradation patterns before rapid capacity loss

## Download Dataset From Here
- https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip