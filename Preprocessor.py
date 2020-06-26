import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import  PCA
import numpy as np
import matplotlib.pyplot as plt

class preprocessingData():
    def __init__(self):
        self.data = pd.read_csv("diabetes.csv")
        self.num_cols = self.data.columns[:len(self.data.columns)-1]
        self.label = self.data.columns[len(self.data.columns)-1]

    def getData(self):
        return self.data

    def getNumData(self):
        return self.data[self.num_cols]

    def getLabelData(self):
        return self.data[self.label]

    def getHeadData(self):
        return self.getNumData().head()

    def getDescribeData(self):
        return self.getNumData().describe()

    def scalerMinMax(self):
        scaler = MinMaxScaler()
        self.data[self.num_cols] = scaler.fit(self.data[self.num_cols]).transform(self.data[self.num_cols])

    def NotOutliers(self):
        Outliers = pd.DataFrame()
        for i in self.num_cols:
            q1,q3 = np.percentile(self.data[i],[25,75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 1.5)
            upper_bound = q3 + (iqr * 1.5)
            Outliers = self.data[(self.data[i] < upper_bound) & (self.data[i] > lower_bound)]
            Outliers.boxplot(column=list(self.num_cols), figsize=(12, 4))

        plt.show()



    def plotBox(self):
        self.data[self.num_cols].boxplot(column=list(self.num_cols),figsize=(12,4))
        plt.show()

    def plotHist(self):
        self.data[self.num_cols].hist(column=list(self.num_cols),figsize=(12,12))
        plt.show()

    def plotLabel(self):
        self.data[self.label].value_counts().plot(kind='bar',figsize=(6,6))
        plt.show()

    def correlation(self):
        return self.data[self.num_cols].corr()


    def plot2DData(self):
        pca = PCA(n_components=2)
        data = pca.fit_transform(self.data[self.num_cols])
        cmp = np.array(['r','g'])
        plt.figure(figsize=(20,6))
        plt.scatter(data[:,0],data[:,1],c=cmp[self.getLabelData().values],s=10, edgecolors='none')
        plt.xlabel("Premier Composant")
        plt.ylabel("Deuxieme Composant")
        plt.title("plot Data")
        plt.show()


















