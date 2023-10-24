import micropip
import pandas as pd
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os

os.environ["OPENAI_API_KEY"] = "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"

ds2022 = pd.read_csv("wusool_trip-2022.csv", usecols=[3,4,5,6])
y2022 = pd.read_csv("wusool_trip-2022.csv", usecols=[10])
ds2021 = pd.read_csv("wusool_trip-2021.csv", usecols=[3,4,5,6])
y2021 = pd.read_csv("wusool_trip-2021.csv", usecols=[10])
ds2020 = pd.read_csv("wusool_trip-2020.csv", usecols=[3,4,5,6])
y2020 = pd.read_csv("wusool_trip-2020.csv", usecols=[10])
ds2019 = pd.read_csv("wusool_trip-2019.csv", usecols=[3,4,5,6])
y2019 = pd.read_csv("wusool_trip-2019.csv", usecols=[10])

ylists = [y2019, y2020, y2021, y2022]
dslists = [ds2019, ds2020, ds2021, ds2022]

y = pd.concat(ylists)
x = pd.concat(dslists)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(xtrain)
xtrain_std = sc.transform(xtrain)
xtest_std = sc.transform(xtest)

model = MLPRegressor(verbose=1, learning_rate_init=0.01)

model.fit(xtrain_std, ytrain)

prediction = model.predict(xtest_std)

np_ytest = ytest.to_numpy()

threshold = 0.8
accuracy = 0
output = ""
for i in range(len(ytest)):
    if np_ytest[i] == 0:
        continue
    percent = prediction[i] / np_ytest[i]
    if percent >= threshold and percent <= 2 - threshold:
        accuracy += 1
        # output = output + "predicted: {}     Actual: {}".format(prediction[i], np_ytest[i])
score = (accuracy / len(ytest))
output = output + "accuracy within {:.0%} of the true value is {:.2%}\n".format(0.8,score)

pyscript.write('trial', output)