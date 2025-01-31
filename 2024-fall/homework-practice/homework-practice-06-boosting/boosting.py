from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = None,
    ):
        self.base_model_class = base_model_class
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)

        self.early_stopping_rounds = early_stopping_rounds

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: z * np.exp(np.log(y))  # Исправьте формулу на правильную. 

    def partial_fit(self, X, y):
        model = self.base_model_class(
            **self.base_model_params).fit(X, y)

        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        if X_val is not None:
            val_predictions = np.zeros(y_val.shape[0])

        bad_rounds_counter = 0

        for i in range(self.n_estimators):

            s = - self.loss_derivative(y_train, train_predictions)

            model = self.partial_fit(X_train, s)

            train_model_predictions = model.predict(X_train)

            gamma = self.find_optimal_gamma(
                y_train, train_predictions, train_model_predictions)

            self.gammas.append(gamma)
            self.models.append(model)

            train_predictions += self.learning_rate * gamma * train_model_predictions
            self.history['train_loss'].append(
                self.loss_fn(y_train, train_predictions))

            self.history['train_roc_auc'].append(
                roc_auc_score(y_train, self.sigmoid(train_predictions)))

            if X_val is not None:
                val_model_predictions = model.predict(X_val)
                val_predictions += self.learning_rate * gamma * val_model_predictions

                self.history['val_loss'].append(
                    self.loss_fn(y_val, val_predictions))
                
                self.history['val_roc_auc'].append(
                    roc_auc_score(y_val, self.sigmoid(val_predictions)))

                if i > 1 and self.history['val_loss'][-1] > self.history['val_loss'][-2]:
                    bad_rounds_counter += 1
                else:
                    bad_rounds_counter = 0

                if self.early_stopping_rounds and bad_rounds_counter > self.early_stopping_rounds:
                    if plot:
                        self.plot_history()

                    return

        if plot:
            self.plot_history()

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])

        for model, gamma in zip(self.models, self.gammas):
            pred += self.learning_rate * gamma * model.predict(X)

        proba = self.sigmoid(pred)
        return np.vstack([1 - proba, proba]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions +
                               gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self):

        if self.history['valid_loss'] is not None:
            history_df = {
                'n_estimators': np.arange(1, len(self.history['train_loss']) + 1),
                'train_loss': self.history['train_loss'],
                'val_loss': self.history['val_loss'],
                'train_roc_auc': self.history['train_roc_auc'],
                'val_roc_auc': self.history['val_roc_auc']
            }

            px.line(history_df, x='n_estimators', y=[
                'train_loss', 'val_loss']).update_yaxes(title_text='loss').show(renderer='png')

            px.line(history_df, x='n_estimators', y=[
                'train_roc_auc', 'val_roc_auc']).update_yaxes(title_text='roc_auc').show(renderer='png')
        else:

            history_df = {
                'n_estimators': np.arange(1,  len(self.history['train_loss']) + 1),
                'train_loss': self.history['train_loss'],
                'train_roc_auc': self.history['train_roc_auc'],
            }

            px.line(history_df, x='n_estimators', y=[
                'train_loss']).update_yaxes(title_text='loss').show(renderer='png')

            px.line(history_df, x='n_estimators', y=[
                'train_roc_auc']).update_yaxes(title_text='roc_auc').show(renderer='png')
