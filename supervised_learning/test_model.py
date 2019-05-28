from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression

X, y = load_iris(return_X_y=True)
y = (y > 0).astype(int) # to binary classes

lr = LogisticRegression(learning_rate=.01, n_iterations=1000)
lr.fit(X, y)
print(lr.predict(X[:2, :]))
print(y[:2])
