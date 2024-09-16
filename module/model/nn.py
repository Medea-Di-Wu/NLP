class NN(object):
    def __init__(self, classes, config):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build()

    def _build(self):
        pass

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.config['epochs'], verbose=True, batch_size=self.config['batch_size'])

    def predict_prob(self, test_x):
        return self.model.predict(test_x)

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5