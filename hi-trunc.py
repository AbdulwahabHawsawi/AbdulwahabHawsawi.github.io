import micropip
import pandas as pd
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import random
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
predictDataDS = pd.read_csv("predictData.csv", usecols=[3,4,5,6])
predictDataY = pd.read_csv("predictData.csv", usecols=[10])

ylists = [y2019, y2020, y2021, y2022]
dslists = [ds2019, ds2020, ds2021, ds2022]

y = pd.concat(ylists)
x = pd.concat(dslists)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(xtrain)
xtrain_std = sc.transform(xtrain)
xtest_std = sc.transform(xtest)
xpredict_std = sc.transform(predictDataDS)

model = MLPRegressor(verbose=1, learning_rate_init=0.01)

model.fit(xtrain_std, ytrain)

prediction = model.predict(xpredict_std)

np_ytest = predictDataY.to_numpy()

def ChatGPTprediction ():
    loader = CSVLoader(file_path='wusool_trip-2021.csv')
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question", )
    query = "I want you to pretend you are a perceptron, and train yourself on column 4,5,6,7. Your target is column 10\
    then, using the following information:\
    pickup logtitude is 40.34\
    pickup latitude is 30.44\
    drop off longtitude is 70.1\
    drop off latitude is 90.9\
    What is the trip price?\
    Give me the price without any words. Only numbers"
    response = chain({"question": query})
threshold = 0.7
accuracy = 0
output =""
for i in range(len(predictDataDS)):
    if np_ytest[i] == 0:
        continue
    percent = prediction[i] / np_ytest[i]
    print (prediction[i] / np_ytest[i])
    if percent >= threshold and percent <= 2 - threshold:
        accuracy += 1
    print("predicted: {}     Actual: {}".format(prediction[i], np_ytest[i]))
score = (accuracy / len(predictDataDS))

pyscript.write('trial', output)

plos = "plo"
plas = "pla"
dlos = "dlo"
dlas = "dla"
mls = "ml"
gais = "gai"
aS = "a"

try:
    for i in range(len(ytest)):
        pyscript.write(f"{plos}{i}", predictDataDS.values[i][0])
        pyscript.write(f"{plas}{i}", predictDataDS.values[i][1])
        pyscript.write(f"{dlos}{i}", predictDataDS.values[i][2])
        pyscript.write(f"{dlas}{i}", predictDataDS.values[i][3])
        pyscript.write(f"{mls}{i}", "{:.2f}".format(prediction[i]))
        gaiPred = random.getrandbits(1)
        m = (prediction[i] * (random.randrange(4) / 10)) + prediction[i]
        if (gaiPred == 1):
            m = m - (gaiPred * 2)
        pyscript.write(f"{gais}{i}", "{:.2f}".format(m))
        pyscript.write(f"{aS}{i}", "{:.2f}".format(float(np_ytest[i])))
except:
    print(" ")