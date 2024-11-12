from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas
  

# Load the dataset
df = pandas.read_csv('normalizedDataset.csv')

# Replace NaN values with the mean of the column
df = df.fillna(df.mean())

# Assume that X is a list of feature vectors and y is a list of labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Create a KFold object with 5 folds
kf = KFold(n_splits=5)

# Initialize lists to store the results for each fold
accuracies = []
geometric_means = []
sensitivities = []
specificities = []

# Split the data into 5 folds and train and evaluate a model for each fold
for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
  
  best_c = None
  best_gamma = None
  best_accuracy = 0

  for c in range(1, 201, 5):
    model = LinearSVC(C=c, max_iter=50000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_c = c

  for gamma in np.arange(0.001, 10.001, 0.5):
        model = SVC(kernel='rbf', C=best_c, gamma=gamma, max_iter=50000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_gamma = gamma
  
  accuracies.append(best_accuracy)


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

print(f"Best C: {best_c}")
print(f"Best_gamma: {best_gamma}")
print(f"Mean accuracy: {mean_accuracy:.2f}")
print(f"Mean geometric mean: {mean_geometric_mean:.2f}")
print(f"Mean sensitivity: {mean_sensitivity:.2f}")
print(f"Mean specificity: {mean_specificity:.2f}")
