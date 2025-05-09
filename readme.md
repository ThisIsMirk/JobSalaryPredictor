# Job Salary Prediction Model

A machine learning project that predicts whether a job will have a high or low salary based on its job description.

## Overview

This project analyzes job posting data to build a classification model that predicts salary levels based solely on the text content of job descriptions. Using natural language processing techniques and various machine learning algorithms, the system identifies patterns in job descriptions that correlate with higher or lower salaries.

## Features

- **Text preprocessing** of job descriptions (lowercasing, removing numbers and special characters)
- **TF-IDF vectorization** with stopword removal to convert text data to numerical features
- **Multiple model comparison** (Multinomial Naïve Bayes, Logistic Regression, Random Forest, SVM)
- **Hyperparameter tuning** via grid search with cross-validation
- **Model evaluation** with accuracy metrics and confusion matrix visualization
- **Feature importance analysis** to identify key words associated with high and low salary jobs

## Dataset

The project uses the "Train_rev1.csv" dataset, which contains job postings with the following key fields:
- `FullDescription`: Text description of the job
- `SalaryNormalized`: Numerical salary value

For efficiency, the analysis uses a random sample of 2,500 job postings.

## Methodology

1. **Data Preprocessing**:
   - Sample 2,500 job postings from the full dataset
   - Clean job descriptions by removing numbers, special characters, and extra spaces
   - Classify jobs as "high salary" (top 25%) or "low salary" (bottom 75%)
   - Split data into training (80%) and testing (20%) sets

2. **Feature Engineering**:
   - Convert text to numerical features using TF-IDF vectorization
   - Extract up to 5,000 features while removing common English stopwords

3. **Model Training and Selection**:
   - Train multiple classification algorithms with grid search for hyperparameter tuning
   - Compare model performance to select the best performer
   - Evaluate final model using confusion matrix and accuracy score

4. **Insights Extraction**:
   - Identify the top 10 words associated with high-salary and low-salary jobs

## Results

The model successfully identifies linguistic patterns in job descriptions that correlate with salary levels. The confusion matrix shows the distribution of correct and incorrect predictions for both high and low salary categories.

### Top Predictive Words

- **High Salary Words**: ['operations' 'architect' 'leadership' 'managing' 'consumer' 'lead' 'head'
 'strategic' 'director' 'senior']
- **Low Salary Words**: ['applicants' 'customer' 'hours' 'training' 'required' 'assistant'
 'standard' 'database' 'basic' 'people']

## Best Model 
The best model is LogisticRegression(C=10, solver='liblinear')
with accuracy of 81%

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
