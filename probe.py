import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def linear_probe(F_train, y_train, F_test, y_test, save_path=None):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(F_train)
    Xte = scaler.transform(F_test)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    clf.fit(Xtr, y_train)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"scaler": scaler, "clf": clf}, f)
    return float(clf.score(Xte, y_test))
