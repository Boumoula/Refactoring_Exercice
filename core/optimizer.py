class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def optimize(self, model):
        print(f"Optimisation terminée (learning_rate = {self.learning_rate})")
