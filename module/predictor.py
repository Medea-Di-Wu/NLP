import csv
import numpy as np
from .calibrator import Calibrator

class Predictor(object):
    def __init__(self, config, logger, model, classes):
        self.config = config
        self.logger = logger
        self.model = model
        self.classes = classes
        if self.config['enable_calibration']:
            self.calibators = []
            for i in range(len(self.classes)):
                self.calibators.append(Calibrator(model_type=self.config['calibrator_type']))

    def predict(self, test_x):
        predictions = self.model.predict(test_x)
        return predictions >= 0.5

    def predict_prob(self, test_x):
        predictions = self.predict_raw_prob(test_x)
        if self.config['enable_calibration']:
            predictions = self._calibrate(predictions)
        return predictions

    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
             # header = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
             header = ['id']
             for cls in self.classes:
             	header.append(cls)
             writer = csv.writer(output_csv_file)
             writer.writerow(header)
             for test_id, prob in zip(test_ids, probs.tolist()):
                 writer.writerow([test_id] + prob)

    def predict_raw_prob(self, test_x):
        predictions = self.model.predict_prob(test_x)
        return predictions

    def _calibrate(self, prob):
        calibrated_prob_list = []
        for i in range(len(self.classes)):
            category = self.classes[i]
            pred_prob = prob[:, i]
            calibrated_prob = self.calibators[i].calibrate(pred_prob)
            calibrated_prob_list.append(calibrated_prob[:, 1])
        calibrated_prob = np.stack(calibrated_prob_list, axis=1)
        return calibrated_prob

    def train_calibrators(self, x, y):
        self.logger.info("Calibators!")
        prob = self.predict_raw_prob(x)  #(batch_size, num_of_cls)
        for i in range(len(self.classes)):
            category = self.classes[i]
            pred_prob = prob[:, i]
            truth_label = y[:, i]
            self.calibators[i].plot_reliability_diagrams(truth_label, pred_prob, category, self.config['calibrators_output_path'])
            uncalibrated_ece, calibrated_ece = self.calibators[i].fit(truth_label, pred_prob)
            self.logger.info("Class: {}, Uncalibrated_ece: {} Calibrated_ece: {}".format(category, uncalibrated_ece, calibrated_ece))