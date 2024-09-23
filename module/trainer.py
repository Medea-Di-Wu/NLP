from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model import NaiveBayes, TextCNN, TextRNN, biLSTM, TransformerClassifier, GRUCNN

class Trainer(object):
    def __init__(self, config, logger, classes, pretrained_embedding):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.pretrained_embedding = pretrained_embedding
        self._create_model(classes)

    def _create_model(self, classes):
        if self.config['model_name'] == 'naivebayes':
            self.model = NaiveBayes(classes)
        elif self.config['model_name'] == 'textcnn':
            self.model = TextCNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'textrnn':
            self.model = TextRNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'biLSTM':
            self.model = biLSTM(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'transformer':
            self.model = TransformerClassifier(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'grucnn':
            self.model = GRUCNN(classes, self.config, self.pretrained_embedding)
        else:
            self.logger.warning("Model Type:{} is not support yet".format(self.config['model_name']))

    def metrics(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def validate(self, validate_x, validate_y):
        predictions = self.model.predict(validate_x)
        return self.metrics(validate_y, predictions)