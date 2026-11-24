from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import SEED

PARAM_GRID = {"clf__n_estimators": [100, 200],
              "clf__max_depth": [None, 20],
              "clf__min_samples_leaf": [1, 3]}


def train_rf(X_train, y_train, X_bal, y_bal, params=None):
    """Random Forest tuned with 5-fold GridSearchCV (scored by ROC AUC); see svm_model."""
    if params:   # quick mode: reuse tuned hyperparameters, no search
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params).fit(X_bal, y_bal)
        return model, {"best_params": dict(params)}
    pipe = Pipeline([("ros", RandomOverSampler(random_state=SEED)),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    search = GridSearchCV(pipe, PARAM_GRID, scoring="roc_auc", n_jobs=1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    search.fit(X_train, y_train)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best).fit(X_bal, y_bal)
    return model, {"best_params": best, "cv_roc_auc": float(search.best_score_)}
