import main
import key
import stock

import os
import time
from datetime import datetime
import joblib
import pytz
import pandas as pd
from keras.models import load_model
from google.cloud import firestore
from google.oauth2 import service_account


# write data into gcp firebase
def writeDataIntoFirebase(thePrice, theDate, compCode="GOOG"):
    """

    :param thePrice:
    :param theDate:
    :param compCode:
    :return:
    """
    credentials = service_account.Credentials.from_service_account_file("key/firebase-admin.json")
    db = firestore.Client(credentials=credentials)
    date_obj = datetime.strptime(theDate, "%Y-%m-%dT%H:%M:%S%z")
    timestamp = int(date_obj.timestamp())

    data = {
        'datatime': timestamp,
        'price': float(thePrice),
        'code': compCode
    }
    collection_name = "classAi_prediction"
    doc_id = None
    result = db.collection(collection_name).document(doc_id).set(data)
    print(result)


# predict next day stocks price.
def predictNextDay(timeStep=60) -> float:
    """
    Predict next day stock's price. Warning: I AM NOT RESPONSIBLE FOR ANYTHING !!!
    :param timeStep: how long the short time memory look back. defaults is 60 days.
    :return: next dat PRICE. it should be a float.
    """
    print("predicting next day")
    # load model
    theModel = load_model('lstm.h5')
    theScaler = joblib.load('scaler.pkl')
    _, data, scaler = main.read_and_preprocess('data/stock_google_new.csv')
    thePrice = main.predict_next_day(model=theModel, scaler=scaler, data=data, time_step=timeStep)
    print(thePrice)
    return thePrice


# do incremental learning. input old model and new data, return new model and upload.
def incremental():
    pass


if __name__ == '__main__':
    # price = predictNextDay(timeStep=60)
    # writeDataIntoFirebase(price, '2024-05-01T03:00:00+08:00')

    nameDate = "2024-05-01"
    nameTime = "T03:00:00+08:00"
    newData = stock.fetch_stock_data('GOOG', nameDate, nameDate)
    print(newData)
    pass
