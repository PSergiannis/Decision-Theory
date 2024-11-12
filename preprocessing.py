import pandas as pd
from sklearn import preprocessing

column_names=["Age","Gender","TB", "DB", "AAP", "ALT", "AST", "TP", "ALB", "A/G", "results"]

df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv', names=column_names)
df["Gender"] = df["Gender"].replace({"Male": 0, "Female": 1})
df["results"] = df["results"].replace({2: 0})


# # Calculate the mean and standard deviation of the data
# mean = df.mean()
# std = df.std()
# # Create a new DataFrame with only the data points that fall within 3 standard deviations of the mean
# df = df[(df - mean).abs() <= 3*std]
# # df = df.dropna()
# df = df.ffill()
# print(std)
# print(mean)
# print(len(df))

# Select all columns except the last one using iloc
data_to_normalize = df.iloc[:, :-1]

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

# Normalize the data
data_normalized = scaler.fit_transform(data_to_normalize)

# Replace the original data with the normalized data
df.iloc[:, :-1] = data_normalized

# print(df) 

df.to_csv("normalizedDataset.csv", index=False)

