from supervised_learning.supervised_learning import SupervisedLearning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class Classification:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __del__(self):
        pass

    def predict_and_analyse(self):
        self.lr_model = LogisticRegression(solver = 'lbfgs')
        self.lr_model.fit(self.x_train, self.y_train)
        self.prediction = self.lr_model.predict(self.x_test)
        print(metrics.confusion_matrix(self.y_test, self.prediction))
