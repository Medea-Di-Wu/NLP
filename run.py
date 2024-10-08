import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')
    logger.info("Start!")

    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)

            logger.info("Preprocessor!")
            preprocessor = Preprocessor(config['preprocessing'], logger)
            data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()
            if config['training']['model_name'] != 'naivebayes':
                config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

            pretrained_embedding = preprocessor.embedding_matrix if config['preprocessing'].get('pretrained_embedding', None) else None

            logger.info("Trainer!")
            trainer = Trainer(config['training'], logger, preprocessor.classes, pretrained_embedding)
            logger.info(f"{train_x.shape=}")
            logger.info(f"{test_x.shape=}")
            model = trainer.fit(train_x, train_y)

            accuracy, cls_report = trainer.validate(validate_x, validate_y)
            logger.info(f"{accuracy=}")
            # logger.info(f"{cls_report=}")
            print(cls_report)

            logger.info("Predictor!")
            predictor = Predictor(config['predict'], logger, model, preprocessor.classes)
            if config['predict']['enable_calibration']:
                predictor.train_calibrators(validate_x, validate_y)
            probs = predictor.predict_prob(test_x)
            predictor.save_result(preprocessor.test_ids, probs)

        except yaml.YAMLError as err:
            logger.warning('Config file err: {}'.format(err))

    logger.info("Completed!")