from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas
df = pandas.read_csv('normalizedDataset.csv')

# replace NaN values with the mean of column 
df = df.fillna(df.mean())

# assume that X is a list of feature vectors and y is a list of labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]



# //////////////////////////////t-test/////////////////////////////////

p_values = []
for feature in X.columns:
    feature_values_disease = X[y==1][feature]
    feature_values_no_disease = X[y==0][feature]
    t, p = ttest_ind(feature_values_disease, feature_values_no_disease)
    p_values.append(p)

# Create a dataframe of features and their corresponding p-values
feature_importance = pandas.DataFrame({'feature': X.columns, 'p_value': p_values})

# Order the features based on p-value
feature_importance = feature_importance.sort_values('p_value')

# Keep the 5 most important features
important_features = feature_importance['feature'].head(5)

# Use only the 5 most important features to train the classifier
X_important = X[important_features]

print(X_important)

# /////////////////////////////////////////////////////////////////////



# create a KFold object with 5 folds
kf = KFold(n_splits=5)

# ... initialize lists to store the results for each fold
accuracies = []
geometric_means = []
sensitivities = []
specificities = []

# split the data into 5 folds and train and evaluate a model for each fold
for train_index, test_index in kf.split(X_important):
  X_train, X_test = X_important.iloc[train_index], X_important.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
  
  model = BernoulliNB()
  model.fit(X_train, y_train)
  
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)

  # Calculate the confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  
  # Calculate the sensitivity and specificity
  tn, fp, fn, tp = cm.ravel()
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)

# Calculate the geometric mean of the sensitivity and specificity
  geometric_mean = np.sqrt(sensitivity * specificity)
  geometric_means.append(geometric_mean)
  sensitivities.append(sensitivity)
  specificities.append(specificity)

# Calculate the mean of the results across all folds
mean_accuracy = np.mean(accuracies)
mean_geometric_mean = np.mean(geometric_means)
mean_sensitivity = np.mean(sensitivities)
mean_specificity = np.mean(specificities)

# Print the results
print(f"Accuracy: {accuracies}")
print(f"Geometric mean: {geometric_means}")
print(f"Sensitivity: {sensitivities}")
print(f"Specificity: {specificities}")

print(f"Mean accuracy: {mean_accuracy:.2f}")
print(f"Mean geometric mean: {mean_geometric_mean:.2f}")
print(f"Mean sensitivity: {mean_sensitivity:.2f}")
print(f"Mean specificity: {mean_specificity:.2f}")
