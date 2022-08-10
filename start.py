import numpy as np

from src import (
    read_config,
    train_lr,
    get_features,
    prepare_data,
    generate_models,
)


def start():
    config = read_config('params.yaml')
    np.random.seed(config['base']['seed'])

    #generate_models(config)
    get_features(config)
    X_train, X_test, y_train, y_test, f_names = prepare_data(config)
    train_lr(X_train, X_test, y_train, y_test, f_names, config)


start()
