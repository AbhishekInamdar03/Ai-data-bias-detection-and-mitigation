# Detecting and Mitigating Bias in Criminal Justice AI (COMPAS Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from aif360.datasets import CompasDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.sklearn.metrics import disparate_impact_ratio
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

import os
import urllib.request

data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'compas')
file_path = os.path.join(data_path, 'compas-scores-two-years.csv')

if not os.path.exists(file_path):
    os.makedirs(data_path, exist_ok=True)
    print("Downloading COMPAS dataset...")
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    urllib.request.urlretrieve(url, file_path)
    print("Download complete.")


# Load and preprocess data
dataset_orig = CompasDataset()

# Split into train and test
train, test = dataset_orig.split([0.8], shuffle=True)

# Baseline model (Logistic Regression)
X_train, y_train = train.features, train.labels.ravel()
X_test, y_test = test.features, test.labels.ravel()

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate accuracy
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))

# Convert predictions into AIF360 dataset for fairness metrics
pred_dataset = test.copy()
pred_dataset.labels = y_pred.reshape(-1, 1)

# Fairness metrics
metric = ClassificationMetric(test, pred_dataset,
                               privileged_groups=[{'race': 1}],
                               unprivileged_groups=[{'race': 0}])
print("Disparate Impact:", metric.disparate_impact())
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())

# Reweighing (Pre-processing)
RW = Reweighing(unprivileged_groups=[{'race': 0}],
                privileged_groups=[{'race': 1}])
train_rw = RW.fit_transform(train)

clf_rw = LogisticRegression(max_iter=1000)
clf_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)
y_pred_rw = clf_rw.predict(X_test)

rw_dataset = test.copy()
rw_dataset.labels = y_pred_rw.reshape(-1, 1)
metric_rw = ClassificationMetric(test, rw_dataset,
                                 privileged_groups=[{'race': 1}],
                                 unprivileged_groups=[{'race': 0}])
print("\nReweighing Accuracy:", accuracy_score(y_test, y_pred_rw))
print("Disparate Impact:", metric_rw.disparate_impact())
print("Equal Opportunity Difference:", metric_rw.equal_opportunity_difference())

# Post-processing (Reject Option Classification)
ROC = RejectOptionClassification(unprivileged_groups=[{'race': 0}],
                                 privileged_groups=[{'race': 1}])
ROC = ROC.fit(test, pred_dataset)
preds_post = ROC.predict(pred_dataset)

metric_post = ClassificationMetric(test, preds_post,
                                   privileged_groups=[{'race': 1}],
                                   unprivileged_groups=[{'race': 0}])
print("\nReject Option Accuracy:", accuracy_score(y_test, preds_post.labels))
print("Disparate Impact:", metric_post.disparate_impact())
print("Equal Opportunity Difference:", metric_post.equal_opportunity_difference())

# Visualization (Bar chart of Equal Opportunity Differences)
labels = ['Baseline', 'Reweighing', 'Reject Option']
eod = [metric.equal_opportunity_difference(),
       metric_rw.equal_opportunity_difference(),
       metric_post.equal_opportunity_difference()]

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=eod)
plt.title("Equal Opportunity Difference across Models")
plt.ylabel("|TPR_priv - TPR_unpriv|")
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()
