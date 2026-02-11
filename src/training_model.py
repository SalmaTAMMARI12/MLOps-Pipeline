from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging

class TrainingStrategy:
    def train(self, X_train, y_train):
        raise NotImplementedError("Chaque stratégie doit implémenter train()")

class LogisticRegressionStrategy(TrainingStrategy):
    def train(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return model

class RandomForestStrategy(TrainingStrategy):
    def train(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        return model
