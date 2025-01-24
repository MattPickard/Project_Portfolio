# Introduction

Welcome to my Turbofan Engine Life Estimations Project! In this project, I built machine learning models to predict airplane engine RUL (Remaining Useful Life) and health status. These types of predictions can be used for scheduling proactive maintenance and ensuring the safety and reliability of the engines. To accomplish this, I utilized run-to-failure engine sensor datasets published by NASA. Run-to-failure datasets are useful for studying the degradation processes of mechanical systems and building models that can monitor and predict such degradation. To make my predictions, I wanted to build models that could successfully extract short-term temporal patterns from 30 seconds of sensor data and use them to make predictions abour RUL and health status. In this context, RUL indicates the number of flight cycles remaining before complete failure, posing a regression prediction task. Health status is a binary classification task, predicting whether the engine is within a normal or abnormal degradation state. NASA observed that all engines experience two phases of degradation, a phase of normal degradation followed by a phase of abnormal degradation before failure.

(Picture of degradation phases)

A depiction that shows an example (variable) of engines, indicating degradation over time. The dashed line indicates the transition from normal to abnormal degradation phases.


# The Data

NASA's Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics: https://www.mdpi.com/2306-5729/6/1/5  
Original 2020 Paper: https://ntrs.nasa.gov/citations/20205001125

These run-to-failure datasets were synthetically generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which simulates turbofan engines with high precision as they are fed flight conditions as inputs as recorded by real comercial jets. The variables used to make the predictions include Flight Data [w] and Sensor Measurments [xs]. In between these two categories, there are 18 features, and each row of data in the dataset represents one second of sensor data.

(variable names)

Each unit (engine) simulated flights of certain lengths and are categorized into three flight classes: short (1 to 3 hour flights), medium (3 to 5 hour flights), and long (5+ hour flights). I tried to include a variety of flight classes to ensure the models would be able to generalize engines from different flight conditions. Below is a table of the 18 units I used to train the models:

<p align="center">
| Dataset | Unit | Flight Class |
|--------------|:-------------:|:--------:|
| DS02-006     | 11 | Short |
| DS02-006     | 14 | Medium |
| DS02-006     | 15, 16, 18, 20 | Long |
| DS03-012     | 1, 5, 9, 12 | Short |
| DS03-012     | 2, 3, 4, 7 | Medium |
| DS03-012     | 6, 8, 10, 11 | Long |
</p>

I used units 13 (Long Flight Class), 14 (Short Flight Class), 15 (Medium Flight Class) from DS03-012 to evaluate model performance.

Due to computational constraints, I limited the scope of the project to a subset of engines that experienced a failure mode that affects both the low pressure turbine and the high pressure turbine efficiency. All of the units used in my project experience this specific type of failure mode. Even while limiting the scope, the raw training data contained over 11 million rows of sensor and flight condition data.  

# Data Preprocessing
Preprocessing Code: 

For the sake of computational efficiency during training, it's usually best to fully preprocess the data to avoid adding computational overhead during training by adding complexity to your data generator. To be compatible as input for the neural networks, the y labels were extracted and the x features were reshaped as (# of samples, 30, 18), representing 30 second windows of 18 features. The 30-second windows were created using overlapping segments, a new window starts every 10 second time steps, meaning there is a slight overlap between each windows. This process converted approximatly 11.5 million seconds of data into 1.15 million 30-second time windows. The 30-second training windows were then randomized and split into training and validation sets, with 10% being used for validation. Here is a list of steps taken to preprocess the data:

1. Extract the correct y labels and x features from the DS02-006 and DS03-012 h5 files and combine them into one dataframe.
2. Split into training and testing sets. (units 13, 14, 15 from DS03-012)
3. Remove Flight Class and Cycle columns. (NASA indicated these were not meant to be used for predictions)
4. Create 30-second windows with 10-second overlaps.
5. Remove all windows that captured data from multiple units. (engines)
6. Remove the Unit column. (this column was needed for the previous step, so it wasn't removed earlier)
7. Randomize the training data and split into training and validation sets.
8. Separate the x features and y labels.
9. Save each set into compressed h5 files for later use.

Due to the size of the dataset, you will see that memory was regularly freed up by deleting variables that were no longer needed after each transformation step. If I had not done this, my computer would have quickly run out of memory.  

# Neural Network Models
Neural Networks Code: 

The first step in building the hybrid models involves building one-dimensional convolutional neural networks (CNNs). These networks are trained, and their convolutional blocks are subsequently utilized as feature extractors for the CatBoost models. I developed these neural networks using TensorFlow and Keras. While I was at it, I decided to take the opportunity to optimize them for the two prediction tasks. While they do not perform as well as the hybrid models, they did pretty well and established a solid baseline and provided scores that serve as benchmarks during evaluation.

Due to the large training dataset, I used a generator that fed the models batches of 30-second windows so I could set custom epoch sizes. With a dataset so large, using the whole dataset as a single epoch would likely mean learning convergence would occur mid-epoch, so I lowered the epoch size to check against the validation set more often. Both neural networks shared a similar structure which I found performed well, which is as follows:

Input shape: (30, 18), thirty seconds of 18 features

One-dimensional Convolutional Block  
    - 1D Convolutional Layer (512 filters, kernel size of 3, strides of 1, relu activation, same padding)  
    - Batch Normalization Layer  
    - Global Average Pooling Layer  
First Dense Block  
    - Dense Layer (2048 units, relu activation, L2 kernel regularization of 0.025)  
    - Batch Normalization Layer  
Eight Smaller Dense Blocks  
    - Dense Layer (128 units, relu activation, L2 kernel regularization of 0.025)  
    - Batch Normalization Layer  
Output Layers  
    - Health State uses a sigmoid activation function. (for binary classification tasks)  
    - RUL uses a linear activation function. (for regression tasks)  

For the optimizer I used the AdamW with an exponential decay learning rate scheduler. This approach allows the learning rate to decrease as the model trains, which promotes more efficient and stable learning. For the health state prediction, I used a simple binary crossentropy loss function, and for the RUL prediction, I used a custom loss function that functions similarly to mean squared error, but penalizes overestimations:



# CatBoost Preprocessing

# CatBoost Models

# Evaluation
Evaluation Code: 


# Next Steps

