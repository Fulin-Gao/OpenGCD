from matplotlib import pyplot as plt
from xgboost import plot_importance
import xgboost as xgb
import numpy as np


def xgboost(x_train, y_train, x_test, y_test, num_classes, plot=False, model=None):

    """
    :param x_train: Training data n1 * m1
    :param y_train: Label n1 * 1 of the training data
    :param x_test: Test data n2 * m2
    :param y_test: Label n2 * 1 of the test data
    :param num_classes: Number of training set classes
    :param plot: Whether to draw feature importance
    :param model: previously trained model
    :return: Probability prediction of test data, label prediction of test data, fitted classifier
    """

    if model is None:
        param = {'objective': 'multi:softprob', 'num_class': num_classes}
        num_round = 50
        xg_train = xgb.DMatrix(x_train, label=y_train)
        xg_test = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(xg_train, 'train')]
        bst = xgb.train(param, xg_train, num_round, watchlist)
        pred_prob = bst.predict(xg_test)
        pred_label = np.argmax(pred_prob, axis=1)
        if plot:
            # Show the importance of features
            plot_importance(bst)
            plt.show()
    else:
        xg_test = xgb.DMatrix(x_test, label=y_test)
        pred_prob = model.predict(xg_test).reshape(y_test.shape[0], num_classes)
        pred_label = np.argmax(pred_prob, axis=1)
        bst = model

    return pred_prob, pred_label, bst
