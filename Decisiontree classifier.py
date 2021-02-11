# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 01:36:48 2019

@author: ASUS
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Social_Network_Ads.csv')

x_set=dataset.iloc[:,1:4].values
y_set=dataset.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
x_set[:,0]=encoder.fit_transform(x_set[:,0])
x_set=np.asarray(x_set,dtype='float32')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_set,y_set,test_size=0.15)

#Initializing the classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)
predictions=predictions.reshape((len(predictions),1))

from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_test,predictions)

print("Accuracy is :",(conf[0][0]+conf[1][1])/predictions.shape[0])

#Visualizing the training set
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
tempx_train=pca.fit_transform(x_train)
tempx_test=pca.fit_transform(x_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
tempx_train = sc.fit_transform(tempx_train)
tempx_test = sc.transform(tempx_test)

classifier1=DecisionTreeClassifier(criterion='entropy')
classifier1.fit(tempx_train,y_train)

from matplotlib.colors import ListedColormap
X_set, Y_set = tempx_train,y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

