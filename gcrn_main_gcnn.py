import sys

import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer
from utils import prepare_dirs, save_config, \
    prepare_config_date, save_results, evaluate_result

config = None


def main(_):

    # Directory generating.. for saving
    prepare_dirs(config)
    prepare_config_date(config, config.ds_ind)
    # Random seed settings
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    # Model training
    trainer = Trainer(config, rng)
    save_config(config.model_dir, config)
    config.load_path = config.model_dir
    if config.is_train:
        trainer.train(save=True)
        result_dict = trainer.test()
    else:
        if not config.load_path:
            raise Exception(
                "[!] You should specify `load_path` to "
                "load a pretrained model")
        result_dict = trainer.test()
    save_results(config.result_dir, result_dict)
    accept_rate = evaluate_result(result_dict, method='KS-test', alpha=0.1)
    kl_div = evaluate_result(result_dict, method='KL')
    wasser_dis = evaluate_result(result_dict, method='wasser')
    sig_test = evaluate_result(result_dict, method='sig_test')
    print("The accept rate of KS test is ", accept_rate)
    print("The final KL div is ", kl_div)
    print("The wasser distance is ", wasser_dis)
    print("The AR of Sign Test is ", sig_test)


if __name__ == "__main__":
    config, unparsed = get_config()
    config.mode = 'prediction'
    config.target = 'hist'
    config.classif_loss = 'kl'
    config.hist_range = list(range(0, 41, 5))

    config.data_rm = 0.5
    config.ds_ind = 0

    # optimal params for kl 1e-3
    config.server_name = 'server_kdd'
    config.conv = 'gcnn'
    config.filter = 'chebyshev5'
    config.is_coarsen = True
    config.is_train = True
    config.stop_early = True
    config.sub_folder = False
    config.stop_win_size = 10
    config.learning_rate = 4e-5
    config.dropout = 0.3
    config.regularization = 1e-4
    config.decay_rate = 0.999
    config.num_kernels = [32, 16]
    config.conv_size = [8, 16]
    config.pool_size = [4, 2]
    config.normalized = True

    # config.server_name = 'server_kdd'
    # config.conv = 'cnn'
    # config.filter = 'conv1'
    # config.is_train = True
    # config.stop_early = True
    # config.sub_folder = False
    # config.learning_rate = 0.00013
    # config.regularization = 0.00057
    # config.drop_out = 0.32
    # config.decay_rate = 0.983
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]

    # config.server_name = 'chengdu'
    # config.conv = 'cnn'
    # config.filter = 'conv1'
    # config.is_train = True
    # config.stop_early = True
    # config.num_epochs = 200
    # config.win_size = 10
    # config.sub_folder = False
    # config.learning_rate = 6.6e-4
    # config.regularization = 1e-6
    # config.dropout = 0.0
    # config.decay_rate = 0.9
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]

    # config.server_name = 'chengdu'
    # config.conv = 'cnn'
    # config.filter = 'conv1'
    # config.is_train = True
    # config.stop_early = True
    # config.num_epochs = 200
    # config.win_size = 10
    # config.sub_folder = False
    # config.learning_rate = 6.6e-3
    # config.regularization = 2e-6
    # config.dropout = 0.35
    # config.decay_rate = 0.99
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 8]
    # config.pool_size = [4, 2]

    # config.server_name = 'chengdu'
    # config.conv = 'gcnn'
    # config.filter = 'chebyshev5'
    # config.is_coarsen = True
    # config.is_train = True
    # config.stop_early = True
    # config.num_epochs = 200
    # config.sub_folder = False
    # config.stop_win_size = 10
    # config.learning_rate = 0.001
    # config.dropout = 0.0
    # config.regularization = 3.32e-5
    # config.decay_rate = 0.95
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]
    # config.normalized = True

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
