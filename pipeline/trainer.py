from core.dataset import PatientDataset
from core.logistic_regression import LogisticRegressionModel

class Trainer:
    def __init__(self, data_path):
        self.dataset = PatientDataset(data_path)
        self.model = LogisticRegressionModel()

    def train_model(self):
        self.dataset.load()
        X, y = self.dataset.get_features_and_labels()

        self.model.train(X, y)
        return self.model


if __name__ == "__main__":
    trainer = Trainer("data/patients.csv")
    trained_model = trainer.train_model()
    print("✅ Modèle entraîné avec succès !")
