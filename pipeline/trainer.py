from core.dataset import PatientDataset
from core.random_forest_model import RandomForestVirusModel

class Trainer:
    def __init__(self, data_path):
        self.dataset = PatientDataset(data_path)
        self.model = RandomForestVirusModel()

    def train_model(self):
        self.dataset.load()
        X, y = self.dataset.get_features_and_labels()

        self.model.train(X, y)  # ⬅️ maintenant ça existe ✅
        self.model.save("data/random_forest_model.joblib")
        return self.model


if __name__ == "__main__":
    trainer = Trainer("data/patients_infectes.csv")
    trained_model = trainer.train_model()
    print("✅ Modèle RandomForest entraîné avec succès !")
