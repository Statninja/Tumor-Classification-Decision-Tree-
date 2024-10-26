
# Classification of Cancer Tumors: Benign or Malignant Using Decision Tree

This project is a comprehensive classification analysis on breast cancer data, focusing on differentiating between benign and malignant tumors using a Decision Tree Classifier. The analysis also includes a simple Decision Tree Regression example, demonstrating the decision tree's ability to handle regression tasks.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Libraries Used](#libraries-used)
3. [Dataset](#dataset)
4. [Feature Engineering and Scaling](#feature-engineering-and-scaling)
5. [Classification Model](#classification-model)
6. [Regression Model](#regression-model)
7. [Results and Evaluation](#results-and-evaluation)
8. [Visualization](#visualization)

## Project Overview
This repository provides a guide for building a Decision Tree model to classify samples as benign or malignant based on key tumor features. The model leverages the Gini impurity criterion to decide split points within the tree and is fine-tuned using hyperparameters to avoid overfitting. Additionally, we use a simple regression model on one feature for illustration purposes.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
  - `DecisionTreeClassifier`
  - `DecisionTreeRegressor`
  - `train_test_split`
  - `StandardScaler`
  - `classification_report`
  - `confusion_matrix`
  - `plot_tree`

## Dataset
The data used is from the `sklearn.datasets` library, specifically the Breast Cancer Wisconsin dataset. It consists of 569 samples with 30 numeric features describing tumor characteristics. The `target` column indicates whether the tumor is benign (`0`) or malignant (`1`).

## Feature Engineering and Scaling
For classification, we selected the following features:
- `mean radius`
- `mean texture`
- `mean perimeter`
- `mean area`
- `mean concavity`
- `mean compactness`
- `worst radius`

The selected features are standardized using `StandardScaler`, enhancing model performance by normalizing feature values.

```python
# Scaling and splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

