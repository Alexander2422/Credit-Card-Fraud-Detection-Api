import uvicorn
from fastapi import FastAPI
from FrauDetection import FrauDetection
import numpy as np
import pickle5
import pandas as pd


app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle5.load(pickle_in)

@app.get('/')
def index():
    return {'message':"The accuracy of this model is 92%"}



@app.post('/predict')
def predictFraud(data:FrauDetection):
    data = data.dict()

    distanceFromHome=data['distanceFromHome']
    distanceFromLastTransaction=data['distanceFromLastTransaction']
    repeatRetailer=data['repeatRetailer']
    usedChip=data['usedChip']
    usedPinNumber=data['usedPinNumber']
    onlineOrder = data['onlineOrder']

    prediction = classifier.predict([[distanceFromHome,distanceFromLastTransaction,repeatRetailer,usedChip,usedPinNumber,onlineOrder]])
    if(prediction[0]>0.6):
        prediction="Transaction may be fraud"
    else:
        prediction="Transaction doesn't seem to be fraud"
    return {
        'prediction' : prediction
    }


if __name__ == 'main':
    uvicorn.run(app, host='127.0.0.1', port=8000)