# Turbofan Engine Prognostics Models

<p align="center">
  <img src="https://plus.unsplash.com/premium_photo-1679758629409-83446005843c?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YWlycGxhbmUlMjB0dXJib2ZhbiUyMGVuZ2luZXxlbnwwfHwwfHx8MA%3D%3D" alt="Turbofan Engine" style="width: 40%;">
</p>  

## Table of Contents
<table>
  <tr>
    <td><a href="#introduction">Introduction</a><br>
    <a href="#data">Data</a><br>
    <a href="#data-preprocessing">Data Preprocessing</a><br>
    <a href="#neural-network-models">Neural Network Models</a><br>
    <a href="#catboost-preprocessing">CatBoost Preprocessing</a><br>
    <a href="#catboost-models">CatBoost Models</a><br>
    <a href="#evaluation">Evaluation</a><br>
    <a href="#next-steps">Next Steps</a><br>
    <a href="#conclusion">Conclusion</a></td>
  </tr>
</table>

## Introduction
<a name="introduction"></a>

For this project, I built machine learning models to predict airplane engine RUL (Remaining Useful Life) and health status. These predictions are valuable as they enable proactive maintenance scheduling, reducing unexpected downtime and maintenance costs. By monitoring engine health, companies can enhance safety and improve overall fleet reliability, leading to increased customer satisfaction and operational efficiency. I trained these models on run-to-failure datasets published by NASA. Run-to-failure datasets are useful for studying the degradation processes of mechanical systems and building models that can help perform prognostics and diagnostics. The models' raw predictions are based on 30 seconds of sensor data, with a running weighted average applied to the raw predictions to provide more accurate and robust final predictions. 

For this project, RUL refers to the number of flight cycles remaining before complete failure, posing a regression prediction task. Health status is a binary classification task, predicting whether the engine is within a normal or abnormal degradation state. This is in reference to NASA's observation that all engines experience two phases of degradation, a phase of normal degradation followed by a phase of abnormal degradation before failure.

---
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/high_pressure_turbine_efficiency.png?raw=true" alt="High Pressure Turbine Efficiency" style="width: 50%;">
</p>  

*A figure that shows high pressure turbine efficiency over time, indicating engine degradation. The dashed lines show the transition from normal to abnormal degradation phases.*  

---

Traditional machine learning models are typically a first choice for tabular data. However, traditional models tend to be less effective at learning temporal patterns in data. To address the need for models capable of extracting short-term temporal patterns from 30 seconds of sensor data, I developed hybrid models that leverage the strengths of both traditional machine learning techniques and neural networks. The models are built using one-dimensional convolutional neural networks (CNNs) to extract features from the sensor data, then those features are used as input for CatBoost models to make their final predictions. I tried various other approaches, such as forms of Long Short-Term Memory (LSTM), residual network inspired architectures, and using the flattened raw data as input for traditional machine learning models. However, the hybrid models presented in this project performed with higher accuracy.  

## Data
<a name="data"></a>
[**NASA's Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics**](https://www.mdpi.com/2306-5729/6/1/5)  
[**NASA's Original 2020 Paper**](https://ntrs.nasa.gov/citations/20205001125)

These run-to-failure datasets were synthetically generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which simulates turbofan engines with high precision as they are fed flight conditions as recorded by real commercial jets. The variables used to make the predictions include Flight Data (w) and Sensor Measurements (xs). Between these two categories there are 18 features, and each row of data in the dataset represents one second of sensor data.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/flight_data.png?raw=true" alt="Flight Data" style="width: 35%;">
</p>
<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/measurements.png?raw=true" alt="Sensor Measurements" style="width: 35%;">
</p>

Each unit (engine) simulated flights of certain lengths and are categorized into three flight classes: short (1 to 3 hour flights), medium (3 to 5 hour flights), and long (5+ hour flights). A variety of flight classes were included to ensure the models would be able to generalize engines from different flight conditions. Below is a table of the 18 units I used to train the models:

| Dataset | Unit | Flight Class |
|--------------|:-------------:|:--------:|
| DS02-006     | 11 | Short |
| DS02-006     | 14 | Medium |
| DS02-006     | 15, 16, 18, 20 | Long |
| DS03-012     | 1, 5, 9, 12 | Short |
| DS03-012     | 2, 3, 4, 7 | Medium |
| DS03-012     | 6, 8, 10, 11 | Long |  

For evaluation, I used units 13 (Long Flight Class), 14 (Short Flight Class), 15 (Medium Flight Class) from DS03-012, to test the models' ability to generalize to different flight classes.

Due to computational constraints, I limited the scope of the project to a subset of engines that experienced a failure mode that affects both the low pressure turbine and the high pressure turbine efficiency. All of the units used in my project experience this specific type of failure mode. With the 18 engines I used for training, it contained over 11 million rows of sensor and flight condition data.  

## Data Preprocessing
<a name="data-preprocessing"></a>
[**Preprocessing Code**](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Turbofan%20Engine%20Prognostics%20Project/preprocessing.ipynb)

The datasets were preprocessed and fully transformed to avoid additional computational overhead during training. To be compatible as input for the neural networks, the y labels were extracted and the x features were reshaped as (# of samples, 30, 18), representing 30 second windows of 18 features. The 30-second windows were created using overlapping segments with a new window starting every 10 seconds. This process converted approximately 11.5 million seconds of data into 1.15 million 30-second time windows. The 30-second training windows were then randomized and split into training and validation sets, with 10% being used for validation.  

Steps used to preprocess the data:

1. Extract the correct y labels and x features from the DS02-006 and DS03-012 h5 files and combine them into one data frame.
2. Split into training and testing sets. (units 13, 14, 15 from DS03-012)
3. Remove Flight Class and Cycle columns. (NASA indicated these were not meant to be used for predictions)
4. Create 30-second windows with 10-second overlaps.
5. Remove all windows that captured data from multiple units.
6. Remove the Unit column.
7. Randomize the training data and split into training and validation sets.
8. Separate the x features and y labels.
9. Save each dataset as a compressed h5 file for later use.

Due to the size of the dataset, memory was regularly freed by deleting variables that were no longer needed after each transformation step.

## Neural Network Models
<a name="neural-network-models"></a>
[**Neural Networks Code**](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Turbofan%20Engine%20Prognostics%20Project/one_d_conv_models.ipynb)

The first step in assembling the hybrid models involves building one-dimensional convolutional neural networks. While these neural networks train, the first convolutional blocks learn low-level features (one block in this case). These blocks are then separated from the larger models and used as feature extractors for traditional machine learning models such as CatBoost. While I was at it, I optimized the models for the two prediction tasks. They do not perform as well as the finished hybrid models, but they showed promise and established a solid baseline of scores for my hybrid models to compare against.

With a dataset so large, using the whole dataset as a single epoch would likely mean learning convergence would occur mid-epoch, to mitigate this, I lowered the epoch size to check against the validation set more often. To set custom epoch sizes, a data generator was used to feed the models batches of the 30-second windows. Both neural networks shared a similar structure which I found performed well:


- **Input shape:** (30, 18) Thirty seconds of 18 features

- **One-dimensional Convolutional Block**  
    - 1D Convolutional Layer (512 filters, kernel size of 3, strides of 1, relu activation, same padding)  
    - Batch Normalization Layer  
    - Global Average Pooling Layer (I found this worked better than a flattening layer or incremental 1D max pooling layers.)  

- **First Dense Block**  
    - Dense Layer (2048 units, relu activation, L2 kernel regularization of 0.025)  
    - Batch Normalization Layer  

- **Eight Smaller Dense Blocks**  
    - Dense Layer (128 units, relu activation, L2 kernel regularization of 0.025)  
    - Batch Normalization Layer  

- **Output Layers**  
    - Health State uses a sigmoid activation function.  
    - RUL uses a linear activation function.  


For optimizers I used AdamW with an exponential decay learning rate scheduler. This approach allows the learning rate to decrease as the model trains, which promotes more efficient and stable learning. For losses, I used a binary cross-entropy for the health state prediction and a custom loss function for RUL that functions similarly to mean squared error, but penalizes overestimations:

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/custom_loss.png?raw=true" alt="Custom Loss Function" style="width: 50%;">
</p>

The idea behind the custom loss function stems from NASA's evaluation scoring function that slightly penalizes overestimations more than underestimations. This makes sense as overestimations may lead to delayed maintenance and increased costs. By using this loss function, the RUL model did better on NASA's scoring function, however, it performed worse on the root mean squared error metric. Therefore, I used a small penalty weight of .05 to balance performance of the two metrics.

## CatBoost Preprocessing
<a name="catboost-preprocessing"></a>
[**CatBoost Preprocessing Code**](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Turbofan%20Engine%20Prognostics%20Project/catboost_preprocessing.ipynb)

Once the convolutional blocks learned to interpret low-level features, their outputs were used as inputs for CatBoost models. The neural network's first convolutional block takes a shape of (# of samples, 30, 18) as input and outputs a shape of (# of samples, 512). The CatBoost models then use those 512 features to make their predictions. To reduce the computational overhead during training and evaluation, I saved the datasets of features produced by the feature extractors for later use.

## CatBoost Models
<a name="catboost-models"></a>
[**CatBoost Models Code**](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Turbofan%20Engine%20Prognostics%20Project/catboost_models.ipynb)

I began by using grid search cross-validation to find the best parameters for the CatBoost models, however the size of the dataset proved a major challenge, both in terms of memory and computational power. My solution was to use a smaller subset of the dataset during the grid search to gain an intuition for possible best parameters for the larger dataset. During cross-validation, it became clear that deeper trees performed well, however, to keep the timeline of this project reasonable, I limited the depth of the trees to 10. For example, an increase of tree depth from 10 to 11 on the full training set would have added an extra 3-4 hours of training on my personal computer. The final parameters and structure of the models are as follows:

**Health State CatBoost Model:**
- learning rate: 0.1
- depth: 10
- \# of trees: 668
- loss function: Logloss
- Approximate size: 11 MB

**RUL CatBoost Model:**
- learning rate: 0.1
- depth: 10
- \# of trees: 5000
- loss function: RMSE
- Approximate size: 81 MB

## Evaluation
<a name="evaluation"></a>
[**Evaluation Code**](https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Turbofan%20Engine%20Prognostics%20Project/evaluation.ipynb)

To create the final predictions from the raw predictions, I applied a running weighted average of 1500 time steps, which is approximately 4 hours. I also applied a threshold of 0.5 to the weighted health state averages to convert the probabilities into a binary classification. Taking these steps make the predictions more robust and accurate.

To evaluate the performance of the models, I tested them on three units of different flight classes. The three units were 13 (Long Flight Class), 14 (Short Flight Class), and 15 (Medium Flight Class) from DS03-012. For evaluation metrics I used accuracy for the health state predictions and three separate metrics for Remaining Useful Life (RUL) predictions: mean absolute error, root mean squared error, and converted NASA's scoring function into an evaluation metric that penalizes overestimations. 

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/nasa_scoring.png?raw=true" alt="NASA's Evaluation Metric" style="width: 30%;">
</p>

NASA's scoring function is shown above where delta is the difference between the predicted RUL and the actual RUL and alpha is set to 1/13 if the RUL is an underestimate and to 1/10 if the RUL is an overestimate. I converted it into an evaluation metric by taking the mean instead of the sum.

## Unit 13 Evaluation (Long Flight Class)
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_13.png?raw=true" alt="Unit 13 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_13.png?raw=true" alt="Unit 13 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|------------------|--------------------|
| Health State Accuracy         | 93.31%           | 97.00%             |
| RUL MAE                       | 6.18             | 5.68               |
| RUL RMSE                      | 8.25             | 7.50               |
| RUL NASA Evaluation Metric    | 1.79             | 1.68               |

## Unit 14 Evaluation (Short Flight Class)
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_14.png?raw=true" alt="Unit 14 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_14.png?raw=true" alt="Unit 14 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|------------------|--------------------|
| Health State Accuracy         | 93.04%           | 98.96%             |
| RUL MAE                       | 3.73             | 3.46               |
| RUL RMSE                      | 5.42             | 4.48               |
| RUL NASA Evaluation Metric    | 1.55             | 1.46               |

## Unit 15 Evaluation (Medium Flight Class)
<p style="display: flex; align-items: center; justify-content: space-between;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/rul_15.png?raw=true" alt="Unit 15 Evaluation" style="width: 48%;">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/hs_15.png?raw=true" alt="Unit 15 Evaluation" style="width: 48%;">
</p>

| Metric                        | Raw Predictions  | Final Predictions |
|-------------------------------|------------------|--------------------|
| Health State Accuracy         | 95.80%           | 99.56%             |
| RUL MAE                       | 2.55             | 1.90               |
| RUL RMSE                      | 4.07             | 2.77               |
| RUL NASA Evaluation Metric    | 1.29             | 1.19               |

### Evaluation Interpretation

The evaluation results demonstrate significant improvements after applying the final prediction techniques. They also show that the models generalize best to medium flight class engines. Understandable, as the average flight and engine conditions between the three flight classes would most closely resemble the conditions of the medium flight class engines.

## Next Steps   
<a name="next-steps"></a>   
### Create a Diagnostic and Prognostic Suite

The models developed in this project would be as tools to monitor engine health and aid in optimizing maintenance scheduling. However, they don't diagnose the causes of failure. For this reason, I suggest building two additional types of models that would aid in engine diagnostics. First, regression prediction models that predict the health parameters (theta), which are also simulated by the C-MAPSS models. This would give engineers insight into the efficiency and health of the engine's various components.

<p align="center">
  <img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/health_parameters.png?raw=true" alt="Health Parameters" style="width: 35%;">
</p>

Second, create a multi-class classification model that identifies the failure mode. All together, these models would form a diagnostic and prognostic suite that would help engineers diagnose the causes of failure, perform proper maintenance, and schedule maintenance.

### Diversify and Scale Up the Data

As mentioned above, the scope of this project was limited to a single failure mode, so these models serve only as a proof of concept. To ensure the models are industry ready, a scale up of the training dataset would be needed to represent all failure modes. Scaling up the training data would also allow the models to generalize better to the different flight classes. I would expect to see improvements to the long flight and short flight class metrics if the training dataset were larger due to having a wider range of flight and engine conditions to learn from, making the predictions more robust to variations.

### Model Improvements

There are multiple approaches still worth exploring to improve the models' performance: 

- Experiment with cross-validation using deeper trees in the CatBoost models. The depth of the trees were limited to 10 to keep the timeline of this project reasonable.
- Try feature engineering using domain expertise or traditional feature selection techniques, such as using rolling averages or lag features and using the HS model predictions as a feature for the RUL model and vice versa.
- Experiment with different architectures, such as using transformers as feature extractors or using other gradient boosting machine learning models.
- Use a larger training dataset to improve generalization.

## Conclusion
<a name="conclusion"></a>

This project highlights how machine learning can be used to build tools that aid in predictive maintenance and diagnostics. By leveraging sensor data and combining one-dimensional convolutional neural networks with advanced machine learning models like CatBoost, robust models capable of predicting the health and Remaining Useful Life (RUL) of mechanical systems can be built. These predictions allow engineers to proactively identify and address potential issues, reducing unexpected downtime and extending the operational lifespan of engines. This enhances safety and reliability and translates into cost reductions and operational efficiency. I hope you enjoyed, please reach out if you have any questions or comments!
