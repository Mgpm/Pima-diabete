import pandas as pd

class preprocessingData():
    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")