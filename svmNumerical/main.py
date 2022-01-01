import pandas as pd
import numpy as np

# for data visualization
from mlxtend.plotting import plot_learning_curves
import seaborn as sns
import matplotlib.pyplot as plt
# for PCA (feature engineering)
from sklearn.decomposition import PCA
# for data scaling
from sklearn.preprocessing import StandardScaler
# for splitting dataset
from sklearn.model_selection import train_test_split
# for fitting SVM model
from sklearn.svm import SVC
# for displaying evaluation metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data.csv")
df.shape
df.dtypes
df.head()
df.tail()
print(df.dtypes, "\n")
df.describe()  # print table datasets
print(df.describe, "\n")
# loading the predictors into dataframe 'X'
# NOTE: we are not choosing columns - 'id', 'diagnosis', 'Unnamed:32'
X = df.iloc[:, 2:32]
print(X.shape, "\n")
X.head()
# loading target values into dataframe 'y'
y = df.diagnosis
print(y.shape, "\n")
y.head()
# coverting categorical data to numerical data
y_num = pd.get_dummies(y)
y_num.tail()
# /*************************1****************************************/
# use only one column for target value
y = y_num.M
y.tail()
X.corr()
plt.figure(figsize=(21, 14))
sns.heatmap(X.corr(), vmin=0.85, vmax=1, annot=True, cmap='Greens', linewidths=0.5)
# reducing the attributes in X dataframe
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 2 drop the highly correlated columns which are not useful i.e., area, perimeter, perimeter_worst, area_worst, perimeter_se, area_se
X_scaled = pd.DataFrame(X_scaled)
X_scaled_drop = X_scaled.drop(X_scaled.columns[[2, 3, 12, 13, 22, 23]], axis=1)
# 3 apply PCA on scaled data
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(X_scaled_drop)
x_pca = pd.DataFrame(x_pca)
print("Before PCA, X dataframe shape = ", X.shape, "\n After PCA, x_pca dataframe shape = ", x_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
# combine PCA data and target data
# 1 set column names for the dataframe
colnames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'diagnosis']
# target data
diag = df.iloc[:, 1:2]
# /*****************************2************************************/
# combine PCA and target data
Xy = pd.DataFrame(np.hstack([x_pca, diag.values]), columns=colnames)
Xy.head()
# visualize data
sns.lmplot("PC1", "PC2", hue="diagnosis", data=Xy, fit_reg=False, markers=["o", "x"])
plt.show()
# Split data for training and testing
X = (Xy.iloc[:, 0:11]).values
# 75:25 train:test data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("X_train shape ", X_train.shape)
print("y_train shape ", y_train.shape)
print("X_test shape ", X_test.shape)
print("y_test shape ", y_test.shape)
# model fitting
svc = SVC()
svc.fit(X_train, y_train)
# predict values
y_pred_svc = svc.predict(X_test)
y_pred_svc.shape
# print confusion matrix
cm = confusion_matrix(y_test, y_pred_svc)
print("Confusion matrix:\n", cm)
creport = classification_report(y_test, y_pred_svc)
print("Classification report:\n", creport)
# sns.countplot(df['diagnosis'],palette='Paired')
m = plt.hist(df[df["diagnosis"] == "M"].radius_mean, bins=30, fc=(1, 0, 0, 0.5), label="Malignant")
b = plt.hist(df[df["diagnosis"] == "B"].radius_mean, bins=30, fc=(1, 0, 0.5), label="Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
sns.heatmap(cm, annot=True, fmt='.0f', cmap="PuRd")
plt.show()
# plot learning curve
plot_learning_curves(X_train, y_train, X_test, y_test, svc)
plt.show()
