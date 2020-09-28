from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from google.colab import files
files.upload()

df = pd.read_csv('AMZN.csv')
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
df.index.name = 'Date'
df

#If tomorrow's close price is greater than today's close price return 1 else 0
df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df

X = df.iloc[:, 0:df.shape[1]-1].values
Y = df.iloc[:, df.shape[1]-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

tree = DecisionTreeClassifier().fit(X_train, Y_train)
print(tree.score(X_test, Y_test))

tree_predictions = tree.predict(X_test)
print(tree_predictions)

Y_test
