import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

trainset = pd.read_csv("./titanic/train.csv")
testset = pd.read_csv("./titanic/test.csv")

X = trainset.iloc[:, 2:].values
y = trainset.iloc[:, 1].values

pass_id = testset.iloc[:, 0].values

X = np.delete(X, [1, 6, 8], 1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#LogisticRegression 0.77033
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0, max_iter=1000)
# classifier.fit(X, y)

#support Vector Machine 0.76555
# from sklearn.svm import SVC
# classifier = SVC(kernel='linear', random_state=0)
# classifier.fit(X, y)

#k Nearest Neighbor 0.75837
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# classifier.fit(X, y)

#kernel SVM 0.77990
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X, y)

#Bayes Therme 0.37799
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X, y)

#Decision Tree 0.69138
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X, y)

#Random Forest 0.75358
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
# classifier.fit(X, y)

#ann 0.78229

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X, y, batch_size=32, epochs=100)


# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

testset = np.delete(testset, [0, 2, 7, 9], 1)
testset = np.array(ct.transform(testset))
testset = imputer.transform(testset)
testset = sc.transform(testset)

y_pred = (ann.predict(testset) > 0.5)

final = np.concatenate((pass_id.reshape(len(pass_id), 1), y_pred.reshape(len(y_pred), 1)), 1)

df = pd.DataFrame(final)
df.to_csv("./Output.csv", header=['PassengerId', 'Survived'], index=False)