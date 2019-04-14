"""
rna

Usage:
  run-all.py [--blanks] [--mixture] [--augment] [--cutoff] [--units <n>] [--epochs <n>] [--batch <s>]

Options:
  -h --help            Show this screen.
  --blanks             Include blanks in the data
  --mixture            If provided, the mixture data is included
  --augment            If provided, include augmented data
  --cutoff             If provided, use cut-off for preprocessing
  --units <n>          Number of units for each conv/dense layer [default: 100]
  --epochs <n>         Number of epochs used for training [default: 100]
  --batch <s>          Size of each batch during training [default: 16]
"""
import logging
from os.path import join
from typing import Tuple

import yaml
from confidence import load_name
from docopt import docopt
from keras import Model

from ml.generator import generate_data, DataGenerator, EvalGenerator
from ml.model import build_model, compile_model, create_callbacks
from utils.utils import create_logging

LOGDIR = create_logging()
logger = logging.getLogger('main')


def create_model(arguments: dict, config: dict, n_classes: int) -> Model:
    """
    Create keras/tf model based on the number of classes, features and the the number of units in the model

    :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
    :param config: confidence object with specific information regarding the data
    :param n_classes: number of classes in the output layer
    :return: A compiled keras model
    """
    # build model
    model = build_model(units=int(arguments['--units']), n_classes=n_classes, n_features=len(config.columns.prediction))
    # compile model
    compile_model(model)

    return model


def create_generators(arguments: dict, config: dict) -> Tuple[DataGenerator, EvalGenerator]:
    """
    Read in data and create two generators (one for training and one for evaluation/testing)

    :param arguments: arguments as parsed by docopt
    :param config: confidence object with specific information regarding the data
    :return: two DataGenerators, the first containing the train data, the second containing the test data
    """
    # generate data and split into train and test, and return the label encoder for the purpose of
    # converting the output (y) from string to a numeric value
    x_train, y_train, x_test, y_test, label_encoder = generate_data(
        file_s=config.files.single, file_m=config.files.mixture,
        type_col=config.columns.type, rep_col=config.columns.replicate,
        val_col=config.columns.validation, pred_col=config.columns.prediction,
        blank_labels=config.sample_types.blanks, filter_labels=config.sample_types.filter, cut_off=config.cut_off,
        include_blanks=arguments["--blanks"], apply_filter=True, include_mixtures=arguments["--mixture"])

    # init sampling for training
    sampling = {"single": 1,
                "mixture": 1 if arguments["--mixture"] else 0,
                "augment": 2 if arguments["--augment"] else 0}

    # select cut-off is specified
    cut_off = config.cut_off if arguments['--cutoff'] else None

    # init train generator
    train_generator = DataGenerator(x_train, y_train, encoder=label_encoder, blank_labels=config.sample_types.blanks,
                                    n_features=len(config.columns.prediction), sampling=sampling,
                                    batch_size=int(arguments["--batch"]), batches_per_epoch=len(x_train),
                                    cut_off=cut_off)

    # init eval generator
    augmented_samples = len(x_test)//2 if arguments["--augment"] else None

    eval_generator = EvalGenerator(x_test, y_test, encoder=label_encoder, blank_labels=config.sample_types.blanks,
                                   augmented_samples=augmented_samples, n_features=len(config.columns.prediction),
                                   cut_off=cut_off)

    return train_generator, eval_generator


def main(arguments: dict, config: dict, logdir: str) -> None:
    """
    main, compiling and running the model

    :param arguments: arguments as parsed by docopt
    :param config: confidence object with specific information regarding the data
    :param logdir: path that is used to store logging
    """
    # create train and validation genrators
    train_gen, validation_gen = create_generators(arguments, config)
    # create model
    model = create_model(arguments=arguments, config=config, n_classes=train_gen.n_classes)
    # log model
    logger.info("==Model==")
    model.summary(print_fn=lambda x: logger.info(x))
    # create callbacks
    callbacks = create_callbacks(int(arguments['--batch']), validation_gen, logdir)

    # fit model
    model.fit_generator(train_gen, epochs=int(arguments["--epochs"]), validation_data=validation_gen,
                        callbacks=callbacks, verbose=1, shuffle=False)
    # store final model.
    model.save(join(LOGDIR, 'model.hdf5'))


if __name__ == '__main__':
    # Parse command line arguments
    arguments = docopt(__doc__, version='rna 0.2')
    # Add to logging
    logger.info('==Command line arguments==')
    logger.info(yaml.dump(arguments, default_flow_style=False))

    # Read config
    config = load_name('rna')
    # Add to logging
    logger.info('==Configuration==')
    logger.info(yaml.dump(config._source, default_flow_style=False))

    # run main function
    main(arguments, config, LOGDIR)
