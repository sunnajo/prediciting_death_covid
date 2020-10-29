'''This script contains functions for data cleaning'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, f1_score, fbeta_score, confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline for converting queried df
def convert(df):
    '''
    Alters queried dataframe into final dataframe
    '''
    #Convert date columns to datetime objects
    df['cdc_report_dt'] = pd.to_datetime(df['cdc_report_dt'])
    df['pos_spec_dt'] = pd.to_datetime(df['pos_spec_dt'])
    df['onset_dt'] = pd.to_datetime(df['onset_dt'])
    
    # Target: death - binary
    df['death'] = df['death'].map({'Yes': 1, 'No': 0})
    
    # Feature: medical condition - binary
    df['med_cond'] = df['med_cond'].replace({'Yes': 1, 'No': 0, 'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    
    # Feature: hospitalized - binary
    df['hosp'] = df['hosp'].replace({'Yes': 1, 'No': 0, 'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    
    # Feature: icu - binary
    df['icu'] = df['icu'].replace({'Yes': 1, 'No': 0, 'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    
    # Feature: age group - categories/ordinal
    df['age_group'] = df['age_group'].replace({'0 - 9 Years': 0, '10 - 19 Years': 1, '20 - 29 Years': 2, '30 - 39 Years': 3, '40 - 49 Years': 4, '50 - 59 Years': 5, '60 - 69 Years': 6, '70 - 79 Years': 7, '80+ Years': 8, 'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    
    # Feature: sex - dummy variables
    df['sex'] = df['sex'].replace({'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    sex_s = pd.Series(df['sex'])
    sex_dummies = pd.get_dummies(sex_s)
    
    # Feature: race/ethnicity - dummy variables
    df['race_ethnicity'] = df['race_ethnicity'].replace({'Missing': np.nan, 'Unknown': np.nan, 'NA': np.nan})
    race_s = pd.Series(df['race_ethnicity'])
    race_dummies = pd.get_dummies(race_s)
    
    # Concatenate dummies dfs with main df
    df = pd.concat([df, race_dummies], axis=1)
    df.drop(columns={'race_ethnicity'}, inplace=True)
    df = pd.concat([df, sex_dummies], axis=1)
    df.drop(columns={'sex'}, inplace=True)
    
    # Extract month from 'pos_spec_dt' column
    df['month'] = df['pos_spec_dt'].dt.month
    
    # Calculate lag times
    df['onset_pos_lag'] = (df['onset_dt'] - df['pos_spec_dt']).dt.days
    df['pos_onset_lag'] = (df['pos_spec_dt'] - df['onset_dt']).dt.days
    df['cdc_pos_lag'] = (df['cdc_report_dt'] - df['pos_spec_dt']).dt.days
    df['pos_cdc_lag'] = (df['pos_spec_dt'] - df['cdc_report_dt']).dt.days
    df['onset_cdc_lag'] = (df['onset_dt'] - df['cdc_report_dt']).dt.days
    df['cdc_onset_lag'] = (df['cdc_report_dt'] - df['onset_dt']).dt.days
    
    # Re-name 'pos_spec_dt' column to date
    df.rename(columns={'pos_spec_dt': 'date'}, inplace=True)
    
    # Drop 'current_status' column
    df.drop(columns='current_status', inplace=True)
    
    return df