import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style



# read in student grade data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

# print(data.head(50))

# convert to arrays
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)




'''
best = 0 # best score is zero. if we score better than 0, save this as model
for _ in range(30):
    # train test split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # define model, calculate model and print accuracy
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("model.pickle", "wb") as f:
            pickle.dump(linear, f)'''


pickle_in = open("model.pickle", "rb")
linear = pickle.load(pickle_in)



# predict coefficients of linear model
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# print data and predictions side by side for test set
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()





