import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Importing iris dataset
iris_dataset = load_iris()

# Features = X, Target = Y
X = iris_dataset.data
Y = pd.Series(data=iris_dataset.target, name='Target')

# Split dataset for model evaluation
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

# Train model using multi classifier
model = OneVsRestClassifier(SVC())
model.fit(x_train, y_train)

# Predict values
prediction = model.predict(X)
score = model.score(x_test, y_test)

# Show distribution plot
sns.distplot(x=Y, color='g', hist=False)
sns.distplot(x=prediction, color='r', axlabel=f'Histogram: OneVsRestClassifier - score {score}', hist=False)

plt.legend(labels=['Target', 'Predicted'])
plt.show(block=True)
