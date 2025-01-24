# Introduction

In this project, I built machine learning models to predict airplane engine RUL (Remaining Useful Life) and health status. These types of predictions are used for scheduling proactive maintenance and ensuring the safety and reliability of mechanical systems. 
To accomplish this, I utilized run-to-failure sensor datasets provided by NASA. Run-to-failure datasets are useful for studying the degradation processes of mechanical systems and building models that can monitor and predict such degradation. To make my predictions, I wanted to build models that could successfully extract short-term temporal patterns from 30 seconds of sensor data and use them to make predictions abour RUL and health status. In this context, RUL indicates the number of flight cycles remaining before complete failure, posing a regression prediction task. Health status is a binary classification task, predicting whether the engine is within a normal or abnormal degradation state. NASA observed that all engines experience two phases of degradation, a phase of normal degradation followed by a phase of abnormal degradation before failure.

(Picture of degradation phases)
A depiction that shows an example (variable) of engines, indicating degradation over time. The dashed line indicates the transition from normal to abnormal degradation phases.


# The Data

NASA's Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics: https://www.mdpi.com/2306-5729/6/1/5
Original 2020 Paper: https://ntrs.nasa.gov/citations/20205001125

This run-to-failure dataset was synthetically generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which simulates turbofan engines with high precision as they are fed flight conditions as inputs as recorded by real comercial jets. The variables used to make the predictions include Flight Data (w) and Sensor Measurments (xs). In total, there are 18 features. Each row of data in the dataset represents one second of sensor data.

(variable names)

Each unit (engine) simulated flights of certain lengths and are categorized into three flight classes: short (1 to 3 hour flights), medium (3 to 5 hour flights), and long (5+ hour flights). Below is a table that shows the 18 units used to train the models:

| Dataset | Unit | Flight Class |
|--------------|:-------------:|:--------:|
| DS02-006     | 11 | Short |
| DS02-006     | 14 | Medium |
| DS02-006     | 15, 16, 18, 20 | Long |
| DS03-012     | 1, 5, 9, 12 | Short |
| DS03-012     | 2, 3, 4, 7 | Medium |
| DS03-012     | 6, 8, 10, 11 | Long |

I then used units 13 (Long Flight Class), 14 (Short Flight Class), 15 (Medium Flight Class) from DS03-012 to evaluate model performance.


Due to computational constraints, I limited the scope of the project to a subset of engines that experienced a failure mode that affects both the low pressure turbine and the high pressure turbine efficiency. All of the units used in my project experience this specific type of failure mode. Even while limiting the scope, the raw training data contained over 11 million rows of sensor and flight condition data.  

# Data Preprocessing
Preprocessing Code: 

For the sake of computational efficiency during training, when possible, it's usually best to fully preprocess the data instead of adding computational overhead by adding complexity to a data generator. The the y labels were extracted and the x features were reshaped as (# of samples, 30, 18), representing 30 second windows of 18 features, to be used as input. 

The 30-second windows were created using overlapping segments, a new window starts every 10 seconds, meaning there is overlapping data between each window. Doing this converted approximatly 11.5 million seconds of data into 1.15 million 30-second time windows. The 30-second training windows were then randomized and split into training and validation sets, with 10% being used for validation. Here is a list of steps taken to preprocess the data:

1. Extract the correct data from the DS02-006 and DS03-012 h5 files and combine them into one dataframe.
2. Split into training and testing sets (units 13, 14, 15 from DS03-012)
3. Remove Flight Class and Cycle columns (NASA indicated these were not meant to be used for predictions)
4. Create 30-second windows with 10-second overlaps
5. Remove all windows that captured data from multiple units (engines)
6. Remove the Unit column (this column was needed for the previous step, so it wasn't removed earlier)
7. Randomize the training data and split into training and validation sets
8. Separate the x features and y labels
9. Save each set into compressed h5 files for later use

Due to the size of the dataset, you will see that memory was regularly freed up by deleting variables that were no longer needed after each transformation step. If I had not done this, my computer would have quickly run out of memory.  

# Neural Network Models
Neural Networks Code: 

The first step in building the hybrid models is to build one-dimensional convolutional neural networks for their convolutional block that is used to extract features and temporal patterns for the CatBoost models. I built these neural networks with TensorFlow and Keras. While I was at it, I decided to optimize them for the two prediction tasks as a fun exersise. While they do not perform as well as the hybrid models, they did pretty well and proved as a solid baseline and scores "to beat" during evaluation.

Due to the large training dataset, I used a generator that fed the models batches of 30-second windows so I could set custom epoch sizes. With a dataset so large, using the whole dataset as a single epoch would likely mean learning convergence would occur mid-epoch, so I lowered the epoch size to check against the validation set more often. Both neural networks shared a similar structure which I found performed well, which is as follows:

Input shape: (30, 18), thirty seconds of 18 features

One-dimensional Convolutional Block
    - 1D Convolutional Layer (filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
    - Batch Normalization
    - Global Average Pooling
First Dense Block
    - Dense Layer (units=2048, activation='relu', kernel_regularizer=l2(0.025))
    - Batch Normalization
Eight Smaller Dense Blocks
    - Dense Layer (units=128, activation='relu', kernel_regularizer=l2(0.025))
    - Batch Normalization
Output Layers
    - Health State is a binary classification task, so the output layer uses a sigmoid activation function.
    - RUL is a regression task, so the output layer uses a linear activation function.

I used the AdamW optimizer with an ExponentialDecay learning rate scheduler as decreasing the learning rate as the model learns allowed for more effecient and stable learning. For the health state prediction, I used a simple binary crossentropy loss function. For the RUL prediction, I used a custom loss function that functions similarly to mean squared error, but uses a penalty weight to penalize overestimations:
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_{\text{true},i} - y_{\text{pred},i} \right)^2
\]

\[
\text{Penalty} = \frac{1}{n} \sum_{i=1}^n \max\left( y_{\text{pred},i} - y_{\text{true},i}, 0 \right)^2
\]

\[
\text{Total Loss} = \text{MSE} + \text{penalty\_weight} \times \text{Penalty}
\]
----------
\[
\text{MSE} = \frac{1}{n} \sum \left( y_{\text{true}} - y_{\text{pred}} \right)^2
\]

\[
\text{Penalty} = \frac{1}{n} \sum \max\left( y_{\text{pred}} - y_{\text{true}}, 0 \right)^2
\]

\[
\text{Total Loss} = \text{MSE} + w \cdot \text{Penalty}
\]









they are still useful for extracting temporal patterns and features. 

 extract temporal patterns and extract features, those features are then fed to a CatBoost descision tree boosting model to make the final predictions. 

The final models have hybrid achitectures that included a one-dimensional convolutional neural network layer to extract temporal patterns and extract features, those features are then fed to a CatBoost descision tree boosting model to make the final predictions. 


In this project, we adopted a hybrid modeling approach that effectively combines deep learning and gradient boosting techniques to enhance performance for both regression and classification tasks. The architecture begins with a 1D convolutional neural network (CNN) layer, which is specifically designed to extract complex temporal features from the sequential sensor data. This convolutional layer identifies patterns and trends that reflect the engine's operational state over time.

After the convolutional layer, we apply batch normalization to stabilize the learning process. This step ensures that the inputs to the following layers maintain consistent distributions, which not only speeds up training but also improves model performance. Next, a global average pooling layer is used to aggregate the extracted features, reducing dimensionality while retaining the crucial information needed for accurate predictions.

The simplified feature representation generated by the global average pooling layer is then input into CatBoost models, a powerful gradient boosting library recognized for its efficiency and effectiveness in managing structured data. We utilize two separate CatBoost models: one for the regression task of predicting Remaining Useful Life (RUL) and another for the binary classification task of assessing the engine's health status (healthy or faulty). This dual-model configuration allows us to optimize each model specifically for its respective prediction task, thereby improving overall model effectiveness.

# CatBoost Preprocessing

# CatBoost Models

# Evaluation
Evaluation Code: 
To thoroughly evaluate the performance of our models, we implemented a comprehensive evaluation framework that employs various metrics suitable for each task. For RUL prediction, we used regression metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). MAE provides a clear measure of the average size of prediction errors, while RMSE highlights larger errors, giving us insight into the model's precision and reliability.

For the health state classification, we utilized a range of classification metrics, including accuracy, precision, recall, and F1-score. Accuracy reflects the overall correctness of the model's predictions, while precision and recall offer deeper insights into the model's ability to accurately identify positive cases (i.e., faulty engines) and minimize false positives and false negatives, respectively. The F1-score acts as a harmonic mean of precision and recall, providing a balanced metric that considers both the model's sensitivity and specificity.

Our hybrid model showed strong performance across all evaluated metrics. In RUL prediction, it achieved low MAE and RMSE values, indicating high accuracy and consistency in its estimates. For health state classification, the model demonstrated high accuracy, precision, recall, and F1-score, highlighting its effectiveness in reliably identifying engine health states. These results confirm the model's ability to extract meaningful features from sensor data and convert them into actionable predictive insights.

# Next Steps

Building on the successes of our current models, we have identified several opportunities for future enhancements. One key area for improvement is the integration of additional sensor data sources, which could enrich the feature set and potentially reveal new patterns that enhance predictive accuracy. Expanding the dataset to include a variety of operational conditions and different engine types may also lead to more generalized and robust model performance.

Furthermore, we plan to experiment with more advanced feature extraction techniques, such as recurrent neural networks (RNNs) or transformer-based architectures, which could further enhance the models' ability to capture long-term dependencies and complex temporal dynamics in the sensor data. Fine-tuning the hyperparameters of the CatBoost models using methods like grid search or Bayesian optimization may also yield performance improvements, optimizing the models for specific prediction tasks.

We are also considering exploring alternative machine learning algorithms, including ensemble methods or hybrid models that combine different types of neural networks, which could provide additional performance benefits. Additionally, implementing real-time data processing pipelines would allow the models to function in live environments, offering continuous monitoring and immediate predictive insights that are invaluable for operational decision-making.

Finally, deploying the models in a production environment and integrating them with existing maintenance management systems would facilitate smooth adoption and utilization. Establishing a feedback loop to incorporate real-world performance data into the training process would support ongoing model refinement and adaptation, ensuring sustained accuracy and relevance in dynamic operational contexts.
