
import sys

import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer
from utils import prepare_dirs, save_config, \
    prepare_config_date, save_results, evaluate_result

config = None


def main(_):

    # for rm in [0.5]:
    #     for ds_ind in range(1):
    for rm in [0.6, 0.7, 0.8]:
        for ds_ind in range(5):
            config.data_rm = rm
            config.ds_ind = ds_ind
            #Directory generating.. for saving
            prepare_dirs(config)
            prepare_config_date(config, config.ds_ind)
            #Random seed settings
            rng = np.random.RandomState(config.random_seed)
            tf.set_random_seed(config.random_seed)

            #Model training
            trainer = Trainer(config, rng)
            save_config(config.model_dir, config)
            config.load_path = config.model_dir
            if config.is_train:
                trainer.train(save=False)
                result_dict = trainer.test()
            else:
                if not config.load_path:
                    raise Exception(
                        "[!] You should specify `load_path` to "
                        "load a pretrained model")
                result_dict = trainer.test()
            save_results(config.result_dir, result_dict)
            accept_rate = evaluate_result(result_dict, method='KS-test', alpha=0.05)
            kl_div = evaluate_result(result_dict, method='KL')
            wasser_dis = evaluate_result(result_dict, method='wasser')
            sig_test = evaluate_result(result_dict, method='sig_test')
            print("The accept rate of KS test is ", accept_rate)
            print("The final KL div is ", kl_div)
            print("The wasser distance is ", wasser_dis)
            print("The AR of Sign Test is ", sig_test)


if __name__ == "__main__":
    config, unparsed = get_config()
    config.mode = 'estimation'
    config.target = 'hist'
    config.classif_loss = 'kl'

    # optimal param for kl 1e-3
    config.server_name = 'server_kdd'
    config.conv = 'cnn'
    config.filter = 'conv1'
    config.is_train = True
    config.stop_early = True
    config.sub_folder = False
    config.learning_rate = 0.001
    config.regularization = 7e-5
    config.drop_out = 0.5
    config.decay_rate = 0.999
    config.num_kernels = [32, 16]
    config.conv_size = [8, 16]
    config.pool_size = [4, 2]

    # config.server_name = 'chengdu'
    # config.conv = 'cnn'
    # config.filter = 'conv1'
    # config.is_train = True
    # config.stop_early = True
    # config.sub_folder = False
    # config.learning_rate = 0.0008
    # config.regularization = 0.005
    # config.drop_out = 0.6
    # config.decay_rate = 0.98
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]

    # config.server_name = 'server_kdd'
    # config.conv = 'gcnn'
    # config.filter = 'chebyshev5'
    # config.is_coarsen = True
    # config.is_train = True
    # config.stop_early = True
    # config.sub_folder = False
    # config.stop_win_size = 10
    # config.learning_rate = 0.002
    # config.dropout = 0.0
    # config.regularization = 5.94e-5
    # config.decay_rate = 1.0
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]
    # config.normalized = True

    # config.server_name = 'server_kdd'
    # config.conv = 'cnn'
    # config.filter = 'conv1'
    # config.is_train = True
    # config.stop_early = True
    # config.sub_folder = False
    # config.learning_rate = 0.01
    # config.regularization = 6.6e-4
    # config.drop_out = 0.0
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
    # config.learning_rate = 6.6e-4
    # config.regularization = 1e-6
    # config.dropout = 0.0
    # config.decay_rate = 0.9
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
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
    # config.learning_rate = 0.018
    # config.dropout = 0.4
    # config.regularization = 3.32e-5
    # config.decay_rate = 0.9286
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 16]
    # config.pool_size = [4, 2]
    # config.normalized = True

    # config.server_name = 'chengdu'
    # config.conv = 'gcnn'
    # config.filter = 'chebyshev5'
    # config.is_coarsen = True
    # config.is_train = True
    # config.stop_early = True
    # config.num_epochs = 200
    # config.sub_folder = False
    # config.stop_win_size = 5
    # config.learning_rate = 4e-3
    # config.dropout = 0.4
    # config.regularization = 2.e-6
    # config.decay_rate = 0.98
    # config.num_kernels = [32, 16]
    # config.conv_size = [8, 8]
    # config.pool_size = [4, 2]
    # config.normalized = True

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
