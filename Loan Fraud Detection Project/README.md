# Loan Fraud Detection
<p align="center">
<img src="" style="width: 40%;">
</p>

## Table of Contents 
<table>
<tr>
<td>
<a href="#introduction">Introduction</a><br>
<a href="#data">Data</a><br>
<a href="#data-preprocessing">Data Preprocessing and Feature Engineering</a><br>
<a href="#sampling">Sampling Strategy</a><br>
<a href="#model-training">LightGBM Model</a><br>
<a href="#evaluation">Evaluation</a><br>
<a href="#next-steps">Next Steps</a><br>
<a href="#conclusion">Conclusion</a>
</td>
</tr>
</table>

## Introduction
<a name="introduction"></a>
For this project, I developed a machine learning model to detect fraudulent loan applications. Early detections of fraudulent activity allows financial institutions to mitigate risks, prevent financial losses, and ensure the integrity of their lending processes. The model uses a combination of applicant information, behavioral data, and derived features to detect potentially fraudulent applications. The challenge in fraud detection often lies not only in achieving high precision, but also doing so with highly imbalanced datasets as fraudent activity occurs within only a small fraction of samples. In loan fraud detection, minimizing false negatives is crucial as it means a legitimate customer is denied credit, so it is common business practice to calibrate the models to perform below a pre-determined False Positive Rate (FPR). I evaluated the model performance at the standard 5% FPR and the more stringent 1% as benchmarks.

## Data
<a name="data"></a>
**Paper:** [Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation](https://arxiv.org/abs/2211.13358)  
**Github:** [Bank Account Fraud](https://github.com/feedzai/bank-account-fraud)

The Bank Account Fraud (BAF) suite of datasets, published at NeurIPS 2022, consists of tabular datasets designed for evaluating machine learning methods in fraud detection. Each dataset was synthetically produced by feeding real bank application data into a CTGAN to preserve the privacy and identities of applicants. For this project I use the base dataset which aims to best represent the orginal bank data, however BAF also features datasets that create different biases within the data. The BAF suite serves as a test bed for assessing both novel and existing machine learning methods.

The base dataset contains 1 million samples of 32 features capturing 8 months of synthetic bank application data. The features were purposly selected by the authors for their predictive power in detecting fraud and there are no missing values. Additionally, The authors of the paper indicate that the first 6 months should be used for training and the last 2 months for testing.

<p align="center">
<img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/fraud_distribution.png?raw=true" style="width: 40%;">
</p>
 <p align="center">As shown above, the dataset exhibits a significant class imbalance with only 1.10% of samples labeled fraudulent.</p>



## Preprocessing and Feature Engineering
<a name="data-preprocessing"></a>
**Code:** [**Preprocessing**](preprocess.ipynb)

The following steps were taken to prepare the data for training:

1. Removed the device_fraud_count feature as there are no positive examples in this dataset.
2. Split the data into training (months 0-5) and testing (months 6-7) sets to evaluate the model's performance on unseen data.
    - After the split, the train set had 794989 samples and the test set had 205010 samples. 
3. Created an income-to-credit-limit ratio feature to capture the relationship between an applicant's income and the proposed credit limit.
4. RobustScaler applied to numerical features to scale while handling outliers.
5. Log scaling of days_since_request, zip_count_4w, and proposed_credit_limit due to skewed distributions to normalize the data. (Q-Q plots shown below)
6. One-hot encoding of categorical features.
7. Memory optimization by downcasting numerical columns to more efficient types.


<p align="center">
<img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/q-q_plots.png?raw=true" style="width: 40%;">
</p>
<p align="center">Q-Q plots are used to compare the distribution to a theoretical normal distribution. The closer the points are to the line, the more normal the distribution.</p>

## Sampling Strategy
<a name="sampling"></a>
**Code:** [**Sampling**](sampling.ipynb)

To address the severe class imbalance, a mix of random undersampling and Synthetic Minority Over-sampling Technique for Nominal and Continuous Features (SMOTENC) oversampling techniques were used to create a balanced training set. This approach yeilded better performance than training on the unsampled imbalanced data using LightGBM's class weight and simple random undersampling. After sampling, the makeup of the 57057 sample training set was as follows:

- **Real Fraudulent Samples:** Constitutes 1/12 of the training set
- **Synthetic Fraudulent Samples:** Generated using SMOTENC, making up 5/12 of the training set
- **Real Non-Fraudulent Samples:** Selected through random undersampling to balance the class distribution

To maximize the real fraudlent class sample representation within the training set, the validation set was created using synthetic positive samples created using SMOTENC. This means that model performce on the validation set will be optimistic, however, it can still provide early stopping feedback to prevent overfitting during training.

## LightGBM Model and Feature Importance
<a name="model-training"></a>
Code: **LightGBM Model**

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. I also tested CatBoost, a similar gradient boosting framework, but I found LightGBM performed slightly better, especially after applying a mixed sampling strategy.

For hyperparameter tuning, Optuna was used to find the best set of hyperparameters to maximize the ROC AUC score. The final training configuration was as follows:

- Objective: "binary"
- Metric: "AUC"
- Num_leaves: 230
- Max_depth: 25
- Learning_rate: 0.071
- Feature_fraction: 0.254

Following training, in-built feature importance metrics can be called to see which features are most influential in the model's predictions. Below I've listed the top 10 features by importance, but the full list is available in the code.

| Rank | Feature | Importance |
|------|--------------------------|------------|
| 1 | velocity_4w | 1937 |
| 2 | days_since_request | 1919 |
| 3 | zip_count_4w | 1638 |
| 4 | velocity_24h | 1542 |
| 5 | velocity_6h | 1531 |
| 6 | name_email_similarity | 1491 |
| 7 | intended_balcon_amount | 1343 |
| 8 | session_length_in_minutes| 1332 |
| 9 | credit_risk_score | 1300 |
| 10 | bank_branch_count_8w | 1174 |


## Evaluation
<a name="evaluation"></a>
Code: **Evaluation**

The model was evaluated using the following metrics: 

### ROC Curve and AUC Score:

<img src="https://github.com/MattPickard/Data-Science-Portfolio/blob/main/Images/roc_curve.png?raw=true" style="width: 40%;">

**ROC AUC Score:** 0.890

Above is the Receiver Operating Characteristic (ROC) curve used to visualize the trade-off between the true positive rate (TPR) and false positive rate (FPR). ROC is usually accompanied by the Area Under the Curve (AUC) score which quantifies the classifier's ability to distinguish between positive and negative classes. An AUC score of 0.5 similar to random guessing, while a score of 1.0 indicates a perfect classifier.

### True Positive Rate at 5% FPR:

**Global #1 Ranked Model from Academic Literature According to paperswithcode.com:** 54.3% ([Paper Link](https://arxiv.org/abs/2401.05240))
**This Model:** 53.93% 

The imbalanced nature of the prediction task means that accuracy is not a good metric to evaluate the model's performance. For example, in a dataset with only 1% fraudlent activity, a model that predicts every transaction as non-fraudulent will achieve an accuracy of 99%. Instead, the True Positive Rate (TPR) at a pre-determined False Positive Rate (FPR) can be used. This metric measures the model's ability to correctly identify fraudulent transactions allowing for a pre-determined rate of false positives. Banks aim to minimize false positives as it may mean denying a loan to a legitimate customer. TPR at 5% FPR is presented as the main metric in the orginal BAF paper to evaluate model performance. 

### True Positive Rate at 1% FPR:

**Global #1 Ranked Model from Academic Literature According to paperswithcode.com:** 25.2% ([Paper Link](https://arxiv.org/abs/2408.12989))
**This Model:** 25.57%

Banks may opt to maintain a a stricter False Positive Rate (FPR) of 1% to minimize the number of legitimate transactions mistakenly identified as fraudulent. At the cost of identifying fewer cases of fraud, this extra level of precision not only fosters trust among customers but also minimizes the resources spent investigating false positives.

### Predictive Equality:

**Model's Predictive Equality at 5% FPR:** 99.49%
**Model's Predictive Equality at 1% FPR:** 99.71%

The authors of the BAF paper also proposed a fairness metric called predictive equality which measures the FPR difference across predetermined groups, in this case age (applicants over 50 vs applicants under 50), where a score of 100% represents perfect equality. The implimentation of such metrics can be useful in identifying model bias and for regulatory compliance. Within BAF suite are various baised datasets which are particularly useful for experimenting with models that aim to minimize bias, however by using the base dataset for this project, achieving a high predictive equality score was trivial.


## Next Steps
<a name="next-steps"></a>
To further enhance the loan fraud detection system, the following steps are recommended:
Model Ensemble:
Combine LightGBM with other models like XGBoost or Random Forest to create an ensemble that can potentially improve performance through diversity.
Advanced Feature Engineering:
Incorporate time-based features or transactional patterns to capture more nuanced behaviors indicative of fraud.
Utilize domain knowledge to create features that reflect common fraud strategies.
Handling Imbalanced Data:
Explore other resampling techniques such as ADASYN or GAN-based methods to generate synthetic samples.
Implement cost-sensitive learning where the model penalizes misclassifications of the minority class more heavily.
Real-Time Deployment:
Integrate the model into a real-time pipeline to flag potentially fraudulent transactions as they occur, enabling immediate action.
Explainability and Interpretability:
Use tools like SHAP or LIME to provide explanations for individual predictions, aiding in trust and transparency with stakeholders.
Continuous Learning:
Implement mechanisms for the model to learn from new data continuously, adapting to evolving fraud patterns.

## Conclusion
<a name="conclusion"></a>
This loan fraud detection project demonstrates the application of machine learning techniques to identify and mitigate fraudulent activities within financial transactions. By meticulously preprocessing the data, optimizing model parameters, and evaluating performance using robust metrics, the developed LightGBM model achieves a high degree of accuracy and fairness. The feature importance analysis provides valuable insights into the factors contributing to fraud, guiding future enhancements and feature engineering efforts. Moving forward, implementing the suggested next steps will further strengthen the model's effectiveness and reliability, ensuring it remains resilient against emerging fraud tactics.

Feel free to explore the provided code notebooks for a deeper understanding of the project's implementation. If you have any questions or feedback, please don't hesitate to reach out!