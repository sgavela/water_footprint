import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from fbprophet import Prophet
import swifter
import pickle

data = pd.read_pickle('df6.pkl')
data['DATE'] = pd.to_datetime(data['DATE'])


def prediction(timestsamp, idx):
    temp_dataf = data[data["ID"] == idx]
    train = temp_dataf[['DATE',"DELTA"]]
    train = train[train['DATE'] <= (timestsamp - timedelta(days = 14) )]
    if train.shape[0] < 7:
        return np.nan
    train.rename(columns = {"DATE":'ds', "DELTA": 'y'}, inplace= True)
    m = Prophet()
    m.fit(train)
    test = m.make_future_dataframe(periods=14)
    forecast = m.predict(test)
    prediction = forecast['yhat'].tail(1)
    return prediction

nex = 0
with open('ids_restantes.pkl', 'rb') as file:
        ids = pickle.load(file)
while len(ids) > 0:
    with open('ids_restantes.pkl', 'rb') as file:
        ids = pickle.load(file)
    nex = ids.pop()
    with open('ids_restantes.pkl', 'wb') as file:
        pickle.dump(ids, file)
    temp_data = data[data['ID'] == nex]
    temp_data['TSPREDICTION'] = temp_data[['DATE', 'ID']].swifter.apply(lambda x: prediction(x[0], x[1]), axis = 1)
    temp_data.to_pickle('data/check'+str(nex)+'.pkl')

print('Terminado')