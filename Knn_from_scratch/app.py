import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from KNeighborsClassifier import KNN

df = pd.read_csv("Social_Network_Ads.csv")
# print(df.head())

df = df.iloc[:,1:]
# print(df.head())

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
# print(df.head())

scaler = StandardScaler()
X = df.iloc[:,0:3]
X = scaler.fit_transform(X)
y = df.iloc[:,-1].values

print(X.shape)
print(y.shape)

X_train, x_test , y_train ,y_test = train_test_split(X,y,test_size=0.2 , random_state=13)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred = knn.predict(x_test)

print(accuracy_score(y_test , y_pred))

score = cross_val_score(knn, X,y,cv=5)
print(score.mean())

apnaKnn = KNN(k=5)

apnaKnn.fit(X_train,y_train)
y_pred1 = apnaKnn.predict(x_test)
print(accuracy_score(y_test,y_pred1))