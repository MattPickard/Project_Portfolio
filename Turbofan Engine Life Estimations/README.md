# Turbofan Engine Prognostics
## Introduction

Welcome to my turbofan engine life estimations project! I will show you how I built machine learning models to predict airplane engine RUL (Remaining Useful Life) and health status. These types of predictions can be used for scheduling proactive maintenance and ensuring the safety and reliability of the engines. To accomplish this, I utilized run-to-failure engine sensor datasets published by NASA. Run-to-failure datasets are useful for studying the degradation processes of mechanical systems and building models that can monitor and predict the failure of systems. My models predict RUL and health status of an engine based on 30 seconds of sensor data. RUL refers to the number of flight cycles remaining before complete failure, posing a regression prediction task. Health status is a binary classification task, predicting whether the engine is within a normal or abnormal degradation state. Health status is in reference to NASA's observation that all engines experience two phases of degradation, a phase of normal degradation followed by a phase of abnormal degradation before failure.


(Picture of degradation phases)

A depiction that shows an example (variable) of engines, indicating degradation over time. The dashed line indicates the transition from normal to abnormal degradation phases.


When it comes to tabular data, traditional machine learning models are usually the first choice. However, these models can struggle to capture temporal patterns in the data. In this case, I needed models that could extract short-term temporal patterns from 30 seconds of sensor data. To do this, I built hybrid models that combined the strengths of both traditional machine learning models and neural networks. They use one-dimensional convolutional neural networks (CNNs) to extract features from the sensor data, then used those features as input for CatBoost models to make their final predictions. I tried various other approaches, such as forms of Long Short-Term Memory (LSTM), residual network inspired architectures, and traditional machine learning models using the raw flattended data. However, I found that these hybrid models performed the best.  

## Data

**NASA's Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics:** https://www.mdpi.com/2306-5729/6/1/5
**NASA's Original 2020 Paper:** https://ntrs.nasa.gov/citations/20205001125

These run-to-failure datasets were synthetically generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which simulates turbofan engines with high precision as they are fed flight conditions as inputs as recorded by real comercial jets. The variables used to make the predictions include Flight Data (w) and Sensor Measurments (xs). Between these two categories there are 18 features, and each row of data in the dataset represents one second of sensor data.

(variable names)

Each unit (engine) simulated flights of certain lengths and are categorized into three flight classes: short (1 to 3 hour flights), medium (3 to 5 hour flights), and long (5+ hour flights). I tried to include a variety of flight classes to ensure the models would be able to generalize engines from different flight conditions. Below is a table of the 18 units I used to train the models:

| Dataset | Unit | Flight Class |
|--------------|:-------------:|:--------:|
| DS02-006     | 11 | Short |
| DS02-006     | 14 | Medium |
| DS02-006     | 15, 16, 18, 20 | Long |
| DS03-012     | 1, 5, 9, 12 | Short |
| DS03-012     | 2, 3, 4, 7 | Medium |
| DS03-012     | 6, 8, 10, 11 | Long |  

For evaluation, I used units 13 (Long Flight Class), 14 (Short Flight Class), 15 (Medium Flight Class) from DS03-012.

Due to computational constraints, I limited the scope of the project to a subset of engines that experienced a failure mode that affects both the low pressure turbine and the high pressure turbine efficiency. All of the units used in my project experience this specific type of failure mode. With the 18 engines I used for training, it contained over 11 million rows of sensor and flight condition data.  

## Data Preprocessing
**Preprocessing Code:** 

I chose to fully preprocess and save the transformed datasets to avoid additional computational overhead during training. To be compatible as input for the neural networks, the y labels were extracted and the x features were reshaped as (# of samples, 30, 18), representing 30 second windows of 18 features. The 30-second windows were created using overlapping segments, a new window starting every 10 seconds, meaning there is a slight overlap between each windows. This process converted approximatly 11.5 million seconds of data into 1.15 million 30-second time windows. The 30-second training windows were then randomized and split into training and validation sets, with 10% being used for validation. Here is a list of steps taken to preprocess the data:

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

## Neural Network Models
**Neural Networks Code:** 

The first step in assembling the hybrid models involves building one-dimensional convolutional neural networks. While these neural networks train, the first convolutional blocks learn low-level features. You can then seperate these blocks from the larger models and use their outputs as feature extractors for traditional machine learning models such as CatBoost. While I was at it, I decided to take the opportunity to optimize the models for the two prediction tasks. They do not perform as well as the finished hybrid models, but they showed promise and established a solid baseline of scores for my hybrid models to compare against.

I used a data generator to feed the models batches of 30-second windows so I could set custom epoch sizes. With a dataset so large, using the whole dataset as a single epoch would likely mean learning convergence would occur mid-epoch, so I lowered the epoch size to check against the validation set more often. Both neural networks shared a similar structure which I found performed well, which is as follows:

**Input shape:** (30, 18) - thirty seconds of 18 features

**One-dimensional Convolutional Block**  
- 1D Convolutional Layer (512 filters, kernel size of 3, strides of 1, relu activation, same padding)  
- Batch Normalization Layer  
- Global Average Pooling Layer (I found this worked better than using a flattening layer or incrimental 1D max pooling layers.)  

**First Dense Block**  
- Dense Layer (2048 units, relu activation, L2 kernel regularization of 0.025)  
- Batch Normalization Layer  

**Eight Smaller Dense Blocks**  
- Dense Layer (128 units, relu activation, L2 kernel regularization of 0.025)  
- Batch Normalization Layer  

**Output Layers**  
- Health State uses a sigmoid activation function.  
- RUL uses a linear activation function.  

For the optimizers I used AdamW with an exponential decay learning rate scheduler. This approach allows the learning rate to decrease as the model trains, which promotes more efficient and stable learning. For losses, I used a binary crossentropy for the health state prediction and a custom loss function for RUL that functions similarly to mean squared error, but penalizes overestimations:

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/custom_loss.png?raw=true" alt="Custom Loss Function" style="width: 50%;">
</p>

The idea behind the custom loss function stems from NASA's evaluation metric that penalizes overestimations. This makes sense as overestimations may lead to late maintenance and are more dangerous. By penalizing overestimations, the RUL model did better on NASA's evaluation metric however it performed worse on the root mean squared error metric. So I landed on using a small penalty weight of .05 to balance performance of the two metrics.

While these models were fun to build, I really only needed their trained convolutional blocks to serve as feature extractors for the CatBoost models. So, let's go to the next step!

## CatBoost Preprocessing
**CatBoost Preprocessing Code:** 

Once the convolutional blocks learned to interprate low-level features, their outputs were used as inputs for CatBoost models. The neural network's first convolutional block takes a shape of (# of samples, 30, 18) as input and outputs a shape of (# of samples, 512). So, the CatBoost models use those 512 features to make their predictions. To reduce the computational overhead during training and evaluation, I saved the datasets of features produced by the feature extractors for later use.

## CatBoost Models
**CatBoost Models Code:** 

I began by using grid search cross-validation to find the best parameters for the CatBoost models, however the size of the dataset proved a major challenge, both in terms of memory and computational power. My solution was to use a smaller subset of the dataset during the grid search to gain an intuition for possible best parameters for the larger dataset. During cross-validation, it became clear that deeper trees performed well, however to keep the timeline of this project reasonable, I limited the final depth of the trees to 10. To give you an example, going from a depth of 10 to 11 on the full training set would have added an extra 3-4 hours of training on my personal computer. The final parameters and structure of the models are as follows:

**Health State CatBoost Model:**
- learning rate: 0.1
- depth: 10
- \# of trees: 668
- loss function: Logloss
- Approximate size with feature extractor: 11 MB

**RUL CatBoost Model:**
- learning rate: 0.1
- depth: 10
- \# of trees: 5000
- loss function: RMSE
- Approximate size with feature extractor: 81 MB

## Evaluation
Evaluation Code: 

To create the final predictions, I applied a running weighted average with windows of 1500 time steps, which is approximately 4 hours, and a threshold of .5 is applied to the health state predictions. This makes the predictions more robust and accurate.

To evaluate the performance of the models, I tested them on three units of different flight classes. The three units were 13 (Long Flight Class), 14 (Short Flight Class), and 15 (Medium Flight Class) from DS03-012. For evaluation metrics, I used accuracy for the health state predictions and three seperate metrics for RUL predictions: mean absolute error, root mean squared error, and NASA's custom evaluation metric that penalizes overestimations. NASA's Scoring Function is shown below where delta is the difference between the predicted RUL and the actual RUL and alpha is set to 1/13 if the RUL is an underestimate and to 1/10 if the RUL is an overestimate. I converted it into an evaluation metric by taking the mean instead of the sum:

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/nasa_scoring.png?raw=true" alt="NASA's Evaluation Metric" style="width: 30%;">
</p>


## Unit 13 Evaluation 
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_13.png?raw=true" alt="Unit 13 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_13.png?raw=true" alt="Unit 13 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|----------------|--------------------------------|
| Health State Accuracy          | 93.31%    | 97.00%                   |
| RUL MAE                       | 6.18      | 5.68                     |
| RUL RMSE                      | 8.25      | 7.50                     |
| RUL NASA Evaluation Metric     | 1.79      | 1.68                     |

## Unit 14 Evaluation 
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_14.png?raw=true" alt="Unit 14 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_14.png?raw=true" alt="Unit 14 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|----------------|--------------------------------|
| Health State Accuracy          | 93.04%    | 98.96%                   |
| RUL MAE                       | 3.73      | 3.46                     |
| RUL RMSE                      | 5.42      | 4.48                     |
| RUL NASA Evaluation Metric     | 1.55      | 1.46                     |

## Unit 15 Evaluation 
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_15.png?raw=true" alt="Unit 15 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_15.png?raw=true" alt="Unit 15 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|----------------|--------------------------------|
| Health State Accuracy          | 95.80%    | 99.56%                   |
| RUL MAE                       | 2.55      | 1.90                     |
| RUL RMSE                      | 4.07      | 2.77                     |
| RUL NASA Evaluation Metric     | 1.29      | 1.19                     |




## Next Steps

More and diverse data, longer time periods, introduce engineered features into the prediction: cycle number, HS prediction, Fc, 5 or 10 minute or flight averages, work with domain experts.  Rolling average of the predictions
Transformer as a feature extractor. 
Experiment with deeper trees in the CatBoost models. 
