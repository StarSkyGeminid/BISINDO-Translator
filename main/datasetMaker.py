import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

class DatasetMaker:
    def __init__(self):
        self.path = 'dataset.csv'
        self.dataFrame = pd.read_csv(self.path)
        self.fit_models = {}
        
    def train(self):
        startline, endline = 52000, 52053

        with open(self.path, 'r', newline='') as f:
            content = [row for i, row in enumerate(csv.reader(f), 1)
                       if i not in range(startline, endline+1)]

        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerows(content)

        X = self.dataFrame.drop('class', axis=1)

        y = self.dataFrame['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=1234)
        
        pipelines = {
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        }
        
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(self.X_train.values, self.y_train.values)
            self.fit_models[algo] = model
            
        for algo, model in self.fit_models.items():
            self.y_prediksi = model.predict(self.X_test.values)
            
        self.export()
        
    def export(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.fit_models['rf'], f)
