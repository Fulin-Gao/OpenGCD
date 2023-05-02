from sklearn.neural_network import MLPClassifier
import numpy as np


def mlp(x_train, y_train, x_test, model=None):
    if model is None:
        clf = MLPClassifier(random_state=1, max_iter=50)
        clf = clf.fit(x_train, y_train)
        pred_prob = clf.predict_proba(x_test)
        pred_label = np.argmax(pred_prob, axis=1)
        model = clf

    else:
        pred_prob = model.predict_proba(x_test)
        pred_label = np.argmax(pred_prob, axis=1)
        model = model

    return pred_prob, pred_label, model