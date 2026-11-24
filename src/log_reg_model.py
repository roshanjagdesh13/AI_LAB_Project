from sklearn.linear_model import LogisticRegression

from src.config import SEED


def train_log_reg(X_bal, y_bal):
    """Linear baseline: how far can a straight decision boundary get us?"""
    return LogisticRegression(max_iter=1000, random_state=SEED).fit(X_bal, y_bal)
