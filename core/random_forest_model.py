from sklearn.ensemble import RandomForestClassifier
import joblib


class RandomForestVirusModel:
    """
    Modèle RandomForest pour le projet virus_diag.
    On garde les mêmes méthodes que LogisticRegressionModel :
    - train(X, y)
    - predict(X)
    - save(path)
    - load(path)
    """

    def __init__(self, n_estimators=100, random_state=42):
        # On crée le modèle sklearn
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """Entraîne le modèle (comme fit)."""
        self.model.fit(X_train, y_train)

    # Facultatif, mais utile si un jour tu l'appelles directement
    def fit(self, X_train, y_train):
        """Alias de train, au cas où."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Retourne les prédictions pour X."""
        return self.model.predict(X)

    def save(self, path: str):
        """Sauvegarde le modèle sur le disque."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Charge un modèle sauvegardé."""
        self.model = joblib.load(path)
