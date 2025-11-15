from sklearn.metrics import log_loss

class LossFunction:
    def compute(self, y_true, y_pred_proba):
        loss = log_loss(y_true, y_pred_proba)
        return loss
