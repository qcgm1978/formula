import pandas
from sklearn import linear_model, tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
class DataTypes(object):
    def __init__(self, n=None):
        if isinstance(n, dict):
            self.info = n
            listProp = list(
                filter(
                    lambda key: isinstance(n[key], (list, np.ndarray)) and key, n.keys()
                )
            )
            if len(listProp):
                self.prop = listProp[0]
                self.list = self[self.prop]
                self.len = len(self.list)
            if 'target' in n:
                self.target=n['target']
            if "file" in n:
                self.df = self.readCsv(n["file"])
                if 'mapData' in n:
                    self.df =self.mapStrToNum(n['mapData'])
        else:
            self.n = n
    def __getitem__(self, i):
        try:
            return self.info[i]
        except KeyError:
            return None
    def getGini(self, getSample=None):
        if callable(getSample):
            samples = getSample(self.df)
        else:
            samples = self.df
            # Gini = 1 - (x/n)2 - (y/n)2
        # Where x is the number of positive answers("GO"), n is the number of samples, and y is the number of negative answers ("NO"), which gives us this calculation:
        n = samples.shape[0]
        x=samples.loc[samples[self.target] == 1].shape[0]
        y = n - x
        Gini = 1 - (x / n) ** 2 - (y / n) ** 2
        return Gini,n,[y,x]
    def getAndFormatData(self, file, dictionary):
        df = self.readCsv(file)
        self.mapStrToNum(dictionary, df)
        return self
    def predictbyDecisionTree(self,features,condition, y=None):
        dtree=self.getDtree(features,y)
        return dtree.predict([condition])
    def getDtree(self, features, y=None):
        if y is None:
            y=self.target
        df = self.df
        X = df[features]
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X, df[y])
        return dtree
    def createDecisionTreeData(self, features, y):
        dtree=self.getDtree(features, y)
        self.graphData = tree.export_graphviz(
            dtree, out_file=None, feature_names=features
        )
        return self
    def graphByData(self, img):
        graph = pydotplus.graph_from_dot_data(self.graphData)
        graph.write_png(img)
        img = pltimg.imread(img)
        imgplot = plt.imshow(img)
        return self
    def pyplot(self, bars=5):
        plt.hist(self.list, bars)
        self.show()
    def polynomialRegressionLine(self):
        x = self.info["x"]
        y = self.info["y"]
        mymodel = np.poly1d(np.polyfit(x, y, 3))
        minX = int(min(x))
        maxX = int(max(x))
        maxY = int(max(y))
        myline = np.linspace(minX, maxX, maxY)
        self.scatter()
        plt.plot(myline, mymodel(myline))
        self.show()
    def predictMultipleRegression(self, file, predictVals):
        X = self.info["x"]
        y = self.info["y"]
        df = self.readCsv(file)
        regr = linear_model.LinearRegression()
        regr.fit(df[X], df[y])
        predict = regr.predict([predictVals])
        return predict[0], list(regr.coef_)
    def predictScale(self, file, toTransformVals):
        df = self.readCsv(file)
        X = df[self.info["x"]]
        y = df[self.info["y"]]
        scale = StandardScaler()
        scaledX = scale.fit_transform(X)
        regr = linear_model.LinearRegression()
        regr.fit(scaledX, y)
        scaled = scale.transform([toTransformVals])
        predict = regr.predict([scaled[0]])
        return predict[0], list(regr.coef_)
    def scale(self, file, scaleCols):
        scale = StandardScaler()
        df = self.readCsv(file)
        X = df[scaleCols]
        scaledX = scale.fit_transform(X)
        return scaledX
    def readCsv(self, file):
        self.df = pandas.read_csv(file)
        return self.df
    def mapStrToNum(self, dictionary, df=None):
        if df is None:
            df = self.df
        for field, v in dictionary.items():
            df[field] = df[field].map(v)
        self.df=df
        return df
    def predictPolynomialRegression(self, predictX):
        mymodel = self.getPolynomialModel()
        return mymodel(predictX)
    def getPolynomialModel(self):
        x = self.info["x"]
        y = self.info["y"]
        mymodel = np.poly1d(np.polyfit(x, y, 4))
        return mymodel
    def getRSquared(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        x, y = self.getData(dataType)
        mymodel = self.getPolynomialModel()
        return r2_score(y, mymodel(x))
    def plotScatter(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        x, y = self.getData(dataType)
        self.scatter(x, y)
        self.show()
    def getData(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        if dataType == "train":
            x = x[:80]
            y = y[:80]
        elif dataType == "test":
            x = x[80:]
            y = y[80:]
        return x, y
    def scatter(self, x=None, y=None):
        if x is None or y is None:
            x = self.info["x"]
            y = self.info["y"]
        plt.scatter(x, y)
    def show(self):
        plt.show()
    def getR(self):
        x = self.info["x"]
        slope, intercept, r, p, std_err = stats.linregress(x, self.info["y"])
        return r
    def predict(self, predictX):
        slope, intercept, r, p, std_err = stats.linregress(
            self.info["x"], self.info["y"]
        )
        return slope * predictX + intercept
    def getModel(self):
        x = self.info["x"]
        myfunc = self.predict
        mymodel = list(map(myfunc, x))
        return mymodel
    def scatterLine(self):
        mymodel = self.getModel()
        self.scatter()
        plt.plot(self.info["x"], mymodel)
        self.show()
    def Numerical(self):
        return self.Discrete() or self.Continuous()
    def Discrete(self):
        return isinstance(self.n, int)
    def Continuous(self):
        return isinstance(self.n, float)
    def Categorical(self):
        return "color"
    def Ordinal(self):
        return "school grades"
    def getMean(self):
        # return sum(self['speed'])/len(self['speed'])
        prop = self.prop
        return np.mean(self[prop])
    def getMedian(self):
        # speed = self['speed'].copy()
        # speed.sort()
        # return speed[len(speed)//2]
        return np.median(self.list)
    def getMode(self):
        return stats.mode(self.list)
    def getStd(self):
        return np.std(self.list)
    def getMSE(self,actual):
        minusSquare = map(lambda x: (x - actual) ** 2, self.list)
        sumVal = sum(list(minusSquare))
        mse=sumVal / self.len
        return mse
    def get1stdProbability(self):
        mean = self.getMean()
        mse=self.getMSE(mean)
        probability = math.sqrt(mse)
        return probability
    def getDistance1std(self):
        expect = self["expectation"]
        if expect:
            mean = self.getMean()
            unitStd = self.get1stdProbability()
            difference = expect - mean
            differenceStd = difference / unitStd
            return differenceStd
        else:
            return self.getStd()
    def getPercentile(self, percent):
        # listP = self.list.copy()
        # listP.sort()
        # lessIndex=round(self.len*percent)
        # val = listP[lessIndex-1]
        # return val
        return np.percentile(self.list, percent * 100)
    def getProbability(self):
        std = self.getDistance1std()
        std2decimal = round(std, 2)
        if std2decimal == 1.00:
            return 0.683
        elif std2decimal == 1.87:
            return 0.015
        elif std2decimal == 2.00:
            return 0.954
    def getVariance(self):
        # mean = self.getMean()
        # difference = map(lambda x: x - mean, self.list)
        # square = map(lambda x: x ** 2, difference)
        # squareList = list(square)
        # variance = sum(squareList) / len(squareList)
        variance = np.var(self.list)
        return variance
