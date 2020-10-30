## Objective

Using machine learning, predict whether or not a patient with laboratory-confirmed COVID-19 will die based on certain patient- and national-level features.

*Disclaimer*

This project is purely for educational purposes. The results or any extensions of this project should NOT be used to guide clinical decision making, personal decisions regarding seeking medical care or treatment, and/or for any other reason(s).



## Table of Contents

- [Data collection and cleaning, EDA, feature analysis and engineering]('https://github.com/sunnajo/predicting_covid_death/tree/main/data-cleaning_analysis')
- [Modeling
- [Modules]('https://github.com/sunnajo/predicting_covid_death/tree/main/modules')
- [Application]('https://github.com/sunnajo/predicting_covid_death/tree/main/application')



## Overview of Data

1) CDC public use surveillance data: patient-level data set that includes demographic and health features, hospitalization and ICU admission status, and outcome (death vs. not)

2) The COVID Tracking Project data: national-level data set that includes total deaths, positive tests, hospitalizations, ICU admissions, patients on ventilatory support, total test results, etc.

3) CDC National Healthcare Safety Network (NHSN) data: state-level data set that includes inpatient hospital bed occupancy by COVID patients and total hospital beds occupied. This data was engineered into a national-level data set.

All data was downloaded directly from the source as .csv files and stored in a PostgreSQL database.

Individual datasets were partially cleaned using SQL with final cleaning and merging done with pandas.



## Methodology/Process

*Data Cleaning/Pre-processing*

- Patient-level data was filtered for those cases that were laboratory confirmed and that had a recorded outcome for death ('Yes' or 'No'). The primary date feature was the date of the positive specimen and the other national-level data sets were merged on this date in an attempt to reflect how COVID-19 was affecting the country as a whole on the date in question. The time period was 4/1/20-9/28/30 as this is the time period that contained the most complete set of data (fewest missing values).
- My final dataset contained 671,435 rows/data points. I engineered several features and ended up with a total of ~40 features prior to my analysis. My target variable was death, 'Yes' or 'No.'
- My dataset contained a fair number of missing values for most of the features. Looking at correlations of the features with my target variable and conducting preliminary feature importance analyses with baseline models of logistic regression and random forest, I was able to determine a multi-pronged strategy for dealing with these missing values.
  - For continuous, numerical, cumulative variables with a time series component (e.g. cumulative deaths), I imputed the value from the day prior. For continuous, numerical variables such as daily increases in certain metrics (e.g. increase in deaths), I imputed the median values of the column.
  - Tackling missing values for the patient-level features that were highly correlated with my target and that appeared to contribute significantly in my classifier model (e.g. hospitalization status, age group) was more difficult. I compared the performance of baseline models (logistic regression, KNN, naive Bayes, decision tree, random forest, XGBoost) using three variations of the data set: 1) imputation of null values using KNN with n_neighbors of 2, 2) simple imputation with a constant value (0.5), 3) all rows/data points with missing values dropped. I oversampled the minority class of my target variable in the training data with RandomOverSampler prior to running these models. The KNN imputation method appeared to perform the best out of these variations. Dropping the null values changed the distribution of my target variable and resulted in lower precision, recall, and F1 scores.
  - Given that XGBoost is able to handle missing values, I also compared the performance of the XGBoost model using two variations of the data set: 1) KNN imputation, 2) null values kept. The models performed similarly.

*Model Selection*

- My target metric was recall as I wanted to try to capture as many actual deaths as possible since missing a death is more dangerous and costly. I secondarily looked at Fbeta scores with beta of 2 to place more weight on recall. I compared baseline models after imputing the remaining null values using KNN with stratified five-fold cross-validation, and logistic regression and XGBoost performed best in terms of recall and Fbeta scores. I then compared the performance of both models trained on imputed training data with that of XGBoost trained on all data with null values kept using stratified five-fold cross-validation and my XGBoost model trained on the pure data performed similarly to that trained on imputed data. I decided to use this model so as to preserve my pure data as imputing patient-level features, especially features like hospitalization or ICU status, is difficult to do accurately and can potentially have dangerous consequences. I also wanted to make my model as generalizable as possible with the assumption that we will likely encounter missing data in this realm.

*Model Optimization*

- Since my target variable classes were imbalanced with a ratio of ~15:1 in favor of the non-death class. I compared several methods for accounting for this imbalance: 1) scaling the weights of the classes with the built-in 'scale_pos_weight' parameter in the XGBoost model, 2) oversampling using RandomOverSampler, SMOTE, ADASYN, and 3) undersampling with RandomUnderSampler. Scaling the weights of the classes with the built-in parameter performed better than random oversampling in my cross-validation comparisons so I decided to use this scaling strategy.
- My baseline XGBoost model was slightly overfit with training test recall score being slightly higher than the validation test recall score on cross-validation. I removed certain features after looking at feature importances, which improved my model's fit. I used RandomizedCVSearch with a target of optimizing recall to look for optimal hyperparameters.



## Results & Conclusions

- My final model included six features: age group, hospitalization, ICU admission, presence of underlying medical condition, national/overall positive rate, male sex. Looking at feature importances, age group, hospitalization, and ICU status were particularly important contributors to my classification model.
- These results make sense from a clinical perspective and correspond with what has already been published by research to date. The potential that the overall positive rate may contribute to the risk of death is an interesting finding and may be an area for future exploration. This was a feature that I had engineered by calculating the proportion of positive test results among total test results and that I did not find commonly reported in publicly available data. It is important to note that this feature likely cannot be taken at face value since it is complex and there may be several confounding variables at play.



## Potential Impacts/Application

- With COVID still at large and hundreds of deaths from COVID occurring every day in the U.S., a machine learning model that is able to predict whether or not a patient with COVID has a high risk of death can have widespread public health implications. In particular, cinicians may be able to use such a method to guide triaging and clinical decision making and hospital administrations may be able use this knowledge to make informed decisions regarding resource utilization.

- I developed an [application]('https://secure-plateau-38454.herokuapp.com/') (for educational purposes only and not to be used in any real-life setting) using Streamlit that would make my model and its results accessible to a user and that also allows for the user to interact with the data sources that I used.




## Future Work

- There is little publicly available/open-source patient-/individual-level data on COVID at this time. This is understandable given the difficulty of COVID data collection, in general, and given the restrictions around protected health information. As more of this data becomes available, it would be interesting to explore other features, especially patient-level features, including clinical characteristics of patients who are hospitalized or require a higher level of care, and geographic features.



## Tools:

- Pandas
- KNNImputer
- Data visualization: matplotlib, seaborn, altair
- Scikit-learn
- Classification models: logistic regression, KNN, naive Bayes, decision trees, random forest, gradient-boosting machines (XGBoost)
- Application: Streamlit, Heroku