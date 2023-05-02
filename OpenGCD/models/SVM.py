from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np


def svm(x_train, y_train, x_test, model=None):
    if model is None:
        clf = make_pipeline(StandardScaler(), SVC(probability=True))
        clf.fit(x_train, y_train)
        pred_prob = clf.predict_proba(x_test)
        pred_label = np.argmax(pred_prob, axis=1)
        model = clf

    else:
        pred_prob = model.predict_proba(x_test)
        pred_label = np.argmax(pred_prob, axis=1)
        model = model

    return pred_prob, pred_label, model