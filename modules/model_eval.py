'''This script contains functions for evaluating models and calculating and visualizing metrics'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, f1_score, fbeta_score, confusion_matrix, classification_report, make_scorer, auc, log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter, OrderedDict
from scipy.stats import randint
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def eval(model_name, model, X, y):
    '''This is a function to compare preliminary models.
    Takes in model and its name from a dictionary containing instantiated models and their names as
    values and keys, respectively, and entire dataframe, partitions data, oversamples minority class
    in training data set, and evaluates metrics'''
    # Partition data
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size = 0.2, random_state=33, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size = 0.2, random_state=14, stratify=y_tv)
    
    # Oversample minority class in training data
    oversample = RandomOverSampler(random_state=0, sampling_strategy='minority')
    X_train_os, y_train_os = oversample.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train_os, y_train_os)
    
    # Make predictions
    y_pred = model.predict(X_val)
    preds = model.predict_proba(X_val)
    
    # Print scores
    print(model_name, ':')
    print('Accuracy score: ', accuracy_score(y_val, y_pred))
    print('Precision score: ', precision_score(y_val, y_pred))
    print('Recall score: ', recall_score(y_val, y_pred))
    print('F1 score: ', f1_score(y_val, y_pred))
    print('F-beta score: ', fbeta_score(y_val, y_pred, beta=2))
    print('ROC-AUC score: ', roc_auc_score(y_val, preds[:,1]), '\n')

def model_scores(model, X, y):
    '''
    Takes in an instantiated model and training data, partitions the training data
    into training and validation sets, trains the model on training data, and returns
    evaluation metrics
    '''
    # Partition data for cross-validation
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make prediction
    y_pred = model.predict(X_val)
    preds = model.predict_proba(X_val)
    
    # Print scores
    print('Accuracy score: ', accuracy_score(y_val, y_pred))
    print('Precision score: ', precision_score(y_val, y_pred))
    print('Recall score: ', recall_score(y_val, y_pred))
    print('F1 score: ', f1_score(y_val, y_pred))
    print('Fbeta score (beta=2): ', fbeta_score(y_val, y_pred, beta=2))
    print('ROC AUC score: ', roc_auc_score(y_val, preds[:,1]), '\n')

def model_scores_os(model, X, y):
    '''
    Takes in an instantiated model and training data, partitions the training data
    into training and validation sets, oversamples the training data, trains the model
    on the oversampled training data, and returns evaluation metrics
    '''
    # Partition data for cross-validation
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    
    # Oversample since classes are imbalanced
    oversampler = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make prediction
    y_pred = model.predict(X_val)
    preds = model.predict_proba(X_val)
    
    # Print scores
    print('Accuracy score: ', accuracy_score(y_val, y_pred))
    print('Precision score: ', precision_score(y_val, y_pred))
    print('Recall score: ', recall_score(y_val, y_pred))
    print('F1 score: ', f1_score(y_val, y_pred))
    print('Fbeta score (beta=2): ', fbeta_score(y_val, y_pred, beta=2))
    print('ROC AUC score: ', roc_auc_score(y_val, preds[:,1]), '\n')

# Plot confusion matrix
def plot_cm(y_test, y_pred):
    '''
    Takes in target variable test set and set of predictions from a model
    and returns confusion matrix
    '''
    # Set up confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
            xticklabels=['No Death', 'Death'],
            yticklabels=['No Death', 'Death'])
    plt.xlabel('Predicted death')
    plt.ylabel('Actual death')
    plt.title('Confusion Matrix')
    plt.show()

# Plot precision-recall curve
def plot_pr_curve(y_test, preds):
    '''
    Takes in target variable test set and set of predictions from a model
    and plots precision-recall curve
    '''
    # Set up precsion-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, preds[:,1])
    
    # Plot P-R curve
    plt.figure(dpi=80, figsize=(5,5))
    plt.plot(thresholds, precision[1:], label='precision')
    plt.plot(thresholds, recall[1:], label='recall')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold')
    plt.title('Precision and Recall Curves')
    plt.show()

# Plot ROC curve and return AUC score
def roc_auc_curve(y_test, preds):
    '''
    Takes in target variable test set and set of predictions from a model,
    plots ROC curve, and prints ROC AUC score
    '''
    # Set up ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, preds[:,1])
    
    # Plot ROC curve
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()
    
    # Print ROC AUC score
    print("ROC AUC score = ", roc_auc_score(y_test, preds[:,1]))

# Cross-validation with stratified KFold (only for models without oversampling)
def cv(model, X_tv, y_tv):
    '''
    Takes in instantiated model and non-test data set, performs cross validation using
    5-fold stratified splits, and returns dataframe of train and test evaluation metrics
    '''
    # Define scoring metrics
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1',
               'fbeta': make_scorer(fbeta_score, beta=2), 'auc': 'roc_auc'}
    
    # Cross-validation using stratified KFolds
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    
    # Store results of cross-validation function dictionary
    cv_dict = cross_validate(model, X_tv, y_tv, scoring=scoring,
                             cv=kf, n_jobs=-1, return_train_score=True)
    
    # Prepare dictionary of metrics for converting into dataframe
    cv_dict_2 = {
        'test_accuracy': np.mean(cv_dict['test_accuracy']),
        'train_accuracy': np.mean(cv_dict['train_accuracy']),
        'test_precision': np.mean(cv_dict['test_precision']),
        'train_precision': np.mean(cv_dict['train_precision']),
        'test_recall': np.mean(cv_dict['test_recall']),
        'train_recall': np.mean(cv_dict['train_recall']),
        'test_f1': np.mean(cv_dict['train_f1']),
        'train_f1': np.mean(cv_dict['test_f1']),
        'test_fbeta': np.mean(cv_dict['train_fbeta']),
        'train_fbeta': np.mean(cv_dict['train_fbeta']),
        'test_auc': np.mean(cv_dict['test_auc']),
        'train_auc': np.mean(cv_dict['train_auc'])
        }
    
    # Convert to dataframe
    cv_df = pd.DataFrame.from_dict(cv_dict_2, orient='index', columns=['mean_score'])
    return cv_df

# Adjust threshold
def threshold(model, X_test, t):
    '''
    Takes in model, val/test data, and a designated threshold value and returns dataframe of
    evaluation metrics based on threshold
    '''
    threshold = t
    y_pred = model.predict(X_test)
    preds = np.where(model.predict_proba(X_test)[:,1] > threshold, 1, 0)
    new_df = pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds), f1_score(y_test, preds), roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "f1", "roc_auc"])
    return new_df

# Look at coefficents and intercept of model
def model_coef(model, X, y):
    # Partition data for cross-validation
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    
    # Oversample since classes are imbalanced
    oversampler = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get coefficients and intercept and format into dataframe
    coef = pd.DataFrame(model.coef_, columns=X_train.columns)
    coef.append(pd.Series(model.intercept_), ignore_index=True)
    return coef.T

def coef_int(model, feat):
    '''
    Takes in model and list containing names of features/columns and returns dataframe of
    coefficients and intercept for model
    '''
    coef = pd.DataFrame(model.coef_, columns=feat)
    coef.append(pd.Series(model.intercept_), ignore_index=True)
    return coef

# Compare sampling methods for a model
def compare_sampling(name, model, X_tv, y_tv):
    '''
    Takes in model and its name (value and key in dictionary of models to compare, respectively)
    and non-test data, splits data into training and validation sets, and trains model on:
    1) non-resampled training data while adjusting built-in class weight metric of model,
    2) training data minority class oversampled using RandomOverSampler,
    3) training data minority class oversampled using SMOTE,
    4) training data minority class oversampled using ADASYN,
    5) training data majority class undersampled using RandomUnderSampler,
    and compares evaluation metrics of these iterations on both training and validation data sets
    '''
    # Partition data
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size = 0.2, random_state=14)
    
    # Model with balancing class weight via model parameter
    if name == 'Random Forest':
        model_balanced = RandomForestClassifier(class_weight='balanced')
    elif name == 'XGBoost':
        model_balanced = XGBClassifier(scale_pos_weight=14)
    
    # Train model
    model_balanced.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train_balanced = model_balanced.predict(X_train)
    y_pred_val_balanced = model_balanced.predict(X_val)
    
    # Store evaluation metrics for train and test sets for balanced model in dictionaries
    balanced_train = {
        'precision': precision_score(y_train, y_pred_train_balanced),
        'recall': recall_score(y_train, y_pred_train_balanced),
        'f1': f1_score(y_train, y_pred_train_balanced),
        'fbeta': fbeta_score(y_train, y_pred_train_balanced, beta=2)
        }
    balanced_val = {
        'precision': precision_score(y_val, y_pred_val_balanced),
        'recall': recall_score(y_val, y_pred_val_balanced),
        'f1': f1_score(y_val, y_pred_val_balanced),
        'fbeta': fbeta_score(y_val, y_pred_val_balanced, beta=2)
        }
    
    # Convert dictionaries to dataframe
    balanced_scores_df = pd.DataFrame({'train': pd.Series(balanced_train), 'val': pd.Series(balanced_val)})
    print('Balanced:')
    print(balanced_scores_df, '\n')
    
    # Models with different sampling methods
    samplers = {
                'Random oversampler': RandomOverSampler(random_state=0, sampling_strategy='minority'),
                'SMOTE': SMOTE(random_state=0, sampling_strategy='minority'),
                'ADASYN': ADASYN(random_state=0, sampling_strategy='minority'),
                'Random undersampler': RandomUnderSampler(random_state=0, sampling_strategy='majority')
               }
    # For each sampling method: resample training data, evalute model for each sampling method
    for name, sampler in samplers.items():
        X_train_rs, y_train_rs = sampler.fit_sample(X_train, y_train)
        model.fit(X_train_rs, y_train_rs)
        y_pred_train, preds_train = model.predict(X_train), model.predict_proba(X_train)
        y_pred_val, preds_val = model.predict(X_val), model.predict_proba(X_val)
        train = {
            'precision': precision_score(y_train, y_pred_train),
            'recall': recall_score(y_train, y_pred_train),
            'f1': f1_score(y_train, y_pred_train),
            'fbeta': fbeta_score(y_train, y_pred_train, beta=2)
            }
        val = {
            'precision': precision_score(y_val, y_pred_val),
            'recall': recall_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val),
            'fbeta': fbeta_score(y_val, y_pred_val, beta=2)
            }
        scores_df = pd.DataFrame({'train': pd.Series(train), 'val': pd.Series(val)})
        print(name, ':')
        print(scores_df, '\n')

# Compare random oversampling and random undersampling methods
def compare_sampling2(model_name, model, X_tv, y_tv):
    '''
    Takes in model and its name (value and key in dictionary of models to compare, respectively)
    and non-test data, splits data into training and validation sets, and trains model on:
    1) non-resampled training data while adjusting built-in class weight metric of model,
    2) training data minority class oversampled using RandomOverSampler,
    3) training data majority class undersampled using RandomUnderSampler,
    and compares evaluation metrics of these iterations on both training and validation data sets
    '''
    # Partition data
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size = 0.2, random_state=14)
    
    # Instantiate & train model with balancing class weight via model parameter
    if model_name == 'Random Forest':
        model_balanced = RandomForestClassifier(class_weight='balanced')
    elif model_name == 'XGBoost':
        model_balanced = XGBClassifier(scale_pos_weight=9)
    model_balanced.fit(X_train, y_train)
    
    # Predictions
    y_pred_train_balanced = model_balanced.predict(X_train)
    y_pred_val_balanced = model_balanced.predict(X_val)
    
    # Dictionaries for scores for class-balanced iteration
    balanced_train = {
        'precision': precision_score(y_train, y_pred_train_balanced),
        'recall': recall_score(y_train, y_pred_train_balanced),
        'f1': f1_score(y_train, y_pred_train_balanced),
        'fbeta': fbeta_score(y_train, y_pred_train_balanced, beta=2)
        }
    balanced_val = {
        'precision': precision_score(y_val, y_pred_val_balanced),
        'recall': recall_score(y_val, y_pred_val_balanced),
        'f1': f1_score(y_val, y_pred_val_balanced),
        'fbeta': fbeta_score(y_val, y_pred_val_balanced, beta=2)
        }
    
    # Convert dictionary to dataframe
    balanced_scores_df = pd.DataFrame({'train': pd.Series(balanced_train), 'val': pd.Series(balanced_val)})
    print('Balanced:')
    print(balanced_scores_df, '\n')
    
    # Models with different sampling methods
    samplers = {
                'Random oversampler': RandomOverSampler(random_state=0, sampling_strategy='minority'),
                'Random undersampler': RandomUnderSampler(random_state=0, sampling_strategy='majority')
               }
    # For each sampling method: resample training data, evalute model for each sampling method
    for name, sampler in samplers.items():
        X_train_rs, y_train_rs = sampler.fit_sample(X_train, y_train)
        model.fit(X_train_rs, y_train_rs)
        y_pred_train, preds_train = model.predict(X_train), model.predict_proba(X_train)
        y_pred_val, preds_val = model.predict(X_val), model.predict_proba(X_val)
        train = {
            'precision': precision_score(y_train, y_pred_train),
            'recall': recall_score(y_train, y_pred_train),
            'f1': f1_score(y_train, y_pred_train),
            'fbeta': fbeta_score(y_train, y_pred_train, beta=2)
            }
        val = {
            'precision': precision_score(y_val, y_pred_val),
            'recall': recall_score(y_val, y_pred_val),
            'f1': f1_score(y_val, y_pred_val),
            'fbeta': fbeta_score(y_val, y_pred_val, beta=2)
            }
        scores_df = pd.DataFrame({'train': pd.Series(train), 'val': pd.Series(val)})
        print(name, ':')
        print(scores_df, '\n')

# Hyperparameter tuning
# (code adapted from https://www.kaggle.com/amypeniston/randomizedsearchcv-for-beginners)
def tune(base_model, parameters, n_iter, kfold, X, y, random_state=42):
    '''
    Takes in model, dictionary of parameters to test, number of iterations, number of folds/splits
    for cross-validation, and non-test data, performs RandomizedSearchCV function, trains model
    on non-test data, and returns mean and std dev of target metric, the best score obtained during
    search, and the corresponding parameters for the model for this best score
    '''
    # Split data into stratified k folds
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
    
    # Instantiate model
    optimal_model = RandomizedSearchCV(base_model,
                            param_distributions=parameters,
                            n_iter=n_iter,
                            cv=k,
                            n_jobs=-1,
                            random_state=42)
    
    # Train model
    optimal_model.fit(X, y)
    
    # Evaluate
    scores = cross_val_score(optimal_model, X, y, cv=k, scoring="recall")    # optimize for recall score
    print("CV mean: {:.3f}, CV std dev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best score: {:.3f}".format(optimal_model.best_score_))
    print("Best parameters: {}".format(optimal_model.best_params_))
    return optimal_model.best_params_, optimal_model.best_score_

# Feature importances
def model_fi(model, X, y):
    '''
    Takes in model and non-test data, partitions data into train/val/test sets, oversamples
    minority class in train set, trains model on oversampled train set, and returns dictionary
    of feature importances for model
    '''
    # Partition data for cross-validation
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    
    # Oversample minority class in training data
    oversampler = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Create list of features and corresponding importances
    feature_list = [list(feature) for feature in zip(X_train.columns, model.feature_importances_)]
    feature_dict = {}
    for f_list in feature_list:
        feature_dict[f_list[0]] = f_list[1]
    
    # Convert list of features and their importances to dictionary
    sorted_dict = sorted(feature_dict.items(), key=operator.itemgetter(1))
    return sorted_dict

def feat_imp(model, X):
    '''
    Takes in model and dataframe excluding target variable, obtains feature importances for model,
    and plots feature importances
    '''
    # Create list of feature name and corresponding feature importance from model
    feature_list = [list(feature) for feature in zip(X.columns, model.feature_importances_)]
    
    # Convert list into dictionary
    feature_dict = {}
    for f_list in feature_list:
        feature_dict[f_list[0]] = f_list[1]
    
    # Convert dictionary into pd.Series for plotting
    feature_s = pd.Series(feature_dict).sort_values(ascending=True)
    
    # Plot feature importances
    feature_s.plot(kind='bar', title='Feature Importance', fontsize=10)
    plt.figure(figsize=(5,4))
    plt.show()