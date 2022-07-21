import os

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np
import pandas as pd
import logging

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

print("Loading model from: {}".format(MODEL_PATH))
clf = load(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ["age","capital.gain","capital.loss","hours.per.week",
                                   "workclass_Federal-gov","workclass_Local-gov","workclass_Private",
                                   "workclass_Self-emp-inc","workclass_Self-emp-not-inc","workclass_State-gov",
                                   "workclass_Without-pay","education_10th","education_11th","education_12th",
                                   "education_1st-4th","education_5th-6th","education_7th-8th","education_9th",
                                   "education_Assoc-acdm","education_Assoc-voc","education_Bachelors",
                                   "education_Doctorate","education_HS-grad","education_Masters","education_Preschool",
                                   "education_Prof-school","education_Some-college","marital.status_Divorced",
                                   "marital.status_Married-AF-spouse","marital.status_Married-civ-spouse",
                                   "marital.status_Married-spouse-absent","marital.status_Never-married",
                                   "marital.status_Separated","marital.status_Widowed","occupation_Adm-clerical",
                                   "occupation_Armed-Forces","occupation_Craft-repair","occupation_Exec-managerial",
                                   "occupation_Farming-fishing","occupation_Handlers-cleaners",
                                   "occupation_Machine-op-inspct","occupation_Other-service","occupation_Priv-house-serv",
                                   "occupation_Prof-specialty","occupation_Protective-serv","occupation_Sales",
                                   "occupation_Tech-support","occupation_Transport-moving","relationship_Husband",
                                   "relationship_Not-in-family","relationship_Other-relative","relationship_Own-child",
                                   "relationship_Unmarried","relationship_Wife","race_Amer-Indian-Eskimo",
                                   "race_Asian-Pac-Islander","race_Black","race_Other","race_White","sex_Female",
                                   "sex_Male","native.country_Cambodia","native.country_Canada","native.country_China",
                                   "native.country_Columbia","native.country_Cuba","native.country_Dominican-Republic",
                                   "native.country_Ecuador","native.country_El-Salvador","native.country_England",
                                   "native.country_France","native.country_Germany","native.country_Greece",
                                   "native.country_Guatemala","native.country_Haiti","native.country_Holand-Netherlands",
                                   "native.country_Honduras","native.country_Hong","native.country_Hungary",
                                   "native.country_India","native.country_Iran","native.country_Ireland",
                                   "native.country_Italy","native.country_Jamaica","native.country_Japan","native.country_Laos",
                                   "native.country_Mexico","native.country_Nicaragua","native.country_Outlying-US(Guam-USVI-etc)",
                                   "native.country_Peru","native.country_Philippines","native.country_Poland",
                                   "native.country_Portugal","native.country_Puerto-Rico","native.country_Scotland",
                                   "native.country_South","native.country_Taiwan","native.country_Thailand",
                                   "native.country_Trinadad&Tobago","native.country_United-States","native.country_Vietnam",
                                   "native.country_Yugoslavia"]
        self.reqparse = reqparse.RequestParser()
        for feature in self._required_features:
            self.reqparse.add_argument(
                feature, type = float, required = True, location = 'json',
                help = 'No {} provided'.format(feature))
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        X = [[args[f]] for f in self._required_features]
        df = pd.DataFrame.from_dict(dict(zip(self._required_features, X)))
        y_pred = clf.predict(df)
        app.logger.info({
            "request_url": request.url,
            "request_remote_addr": request.remote_addr,
            "labels": self._required_features,
            "features": df,
            "prediction": y_pred.tolist()[0]})
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
