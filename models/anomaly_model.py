from sklearn.ensemble import IsolationForest

class ManipulationDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.15,
            random_state=42
        )

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)  # -1 = anomaly
