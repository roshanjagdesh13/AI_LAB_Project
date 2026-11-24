from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.config import SEED

PARAM_GRID = {"clf__C": [1, 10], "clf__gamma": ["scale", 0.3]}


def train_svm(X_train, y_train, X_bal, y_bal, params=None):
    """
    RBF-SVM tuned with 5-fold GridSearchCV (scored by ROC AUC).
    Oversampling happens INSIDE each CV fold (imblearn Pipeline) so duplicated minority
    rows never leak between the fold's train and validation parts. The final model is
    refit with probability=True on the full oversampled training set.
    """
    if params:   # quick mode: reuse tuned hyperparameters, no search
        model = SVC(kernel="rbf", probability=True, random_state=SEED, **params).fit(X_bal, y_bal)
        return model, {"best_params": dict(params)}
    pipe = Pipeline([("ros", RandomOverSampler(random_state=SEED)),
                     ("clf", SVC(kernel="rbf", random_state=SEED))])
    search = GridSearchCV(pipe, PARAM_GRID, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    search.fit(X_train, y_train)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    model = SVC(kernel="rbf", probability=True, random_state=SEED, **best).fit(X_bal, y_bal)
    return model, {"best_params": best, "cv_roc_auc": float(search.best_score_)}
