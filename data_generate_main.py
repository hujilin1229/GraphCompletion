import sys
import os
sys.path.append('../')
import numpy as np
import dataset as ds
import datetime

seed = 123
np.random.seed(seed)
SERVER_NAME = 'server_kdd' # either 'chengdu' or 'server_kdd'
# TARGET = 'avg'
MODE = 'prediction' # either 'estimation' or 'prediction'
TARGET = 'hist' # either 'hist' or 'avg'
SAMPLE_RATE = 15
# Actually, the time window size is (WINDOW_SIZE * SAMPLE_RATE) mins
# the bin of histogram whose unit is m/s = 3.6 km/h
HIST_START = 0
HIST_END = 41
HIST_INTERVAL = 5

CONF_DIR = os.path.join('.', 'conf')
data_rm_list = [0.5, 0.6, 0.7, 0.8]

for SERVER_NAME in ['server_kdd', 'chengdu']:
    for TARGET in ['hist', 'avg']:
        for HIST_INTERVAL in [5]:
            HIST_RANGE = list(range(HIST_START, HIST_END, HIST_INTERVAL))
            if SERVER_NAME == 'server_kdd':
                YEAR = 2016
                WINDOW_SIZE = 2
                Training_start_date = datetime.datetime(YEAR, 7, 19)
                start_months = [7, 8, 8, 9, 10]
                start_dates = [19, 10, 31, 20, 10]
                end_months = [8, 8, 9, 10, 10]
                end_dates = [10, 31, 20, 10, 31]
            elif SERVER_NAME == 'chengdu':
                YEAR = 2014
                WINDOW_SIZE = 3
                Training_start_date = datetime.datetime(YEAR, 8, 2)
                start_months = [8, 8, 8, 8, 8]
                start_dates = [2, 8, 12, 18, 22]
                end_months = [8, 8, 8, 8, 8]
                end_dates = [8, 12, 18, 22, 31]
            else:
                raise Exception("[!] Unkown server name: {}".format(SERVER_NAME))

            for cross_i in range(len(start_months)):
                start_month = start_months[cross_i]
                start_date = start_dates[cross_i]
                end_month = end_months[cross_i]
                end_date = end_dates[cross_i]
                Val_start_date = datetime.datetime(YEAR, start_month, start_date)
                Val_end_date = datetime.datetime(YEAR, end_month, end_date)

                for DATA_RM in data_rm_list:

                    BASE_DIR = os.path.join('.', 'data', SERVER_NAME)
                    if TARGET == 'avg':
                        DIR_DATA = os.path.join(BASE_DIR, '{}_{}'.format(SAMPLE_RATE, WINDOW_SIZE), MODE, TARGET,
                                                '{}_{}-{}_{}'.format(start_date, start_month, end_date, end_month),
                                                'rm{}'.format(DATA_RM))
                    else:
                        DIR_DATA = os.path.join(BASE_DIR, '{}_{}'.format(SAMPLE_RATE, WINDOW_SIZE), MODE, TARGET,
                                                '{}_{}_{}'.format(HIST_START, HIST_END, HIST_INTERVAL),
                                                '{}_{}-{}_{}'.format(start_date, start_month, end_date, end_month),
                                                'rm{}'.format(DATA_RM))

                    try:
                        os.stat(DIR_DATA)
                    except:
                        os.makedirs(DIR_DATA)

                    CAT_HEAD = []  # ['time_index', 'dayofweek']
                    CON_HEAD = []
                    prep_param = {'data_dir': DIR_DATA,
                                  'base_dir': BASE_DIR,
                                  'server_name': SERVER_NAME,
                                  'conf_dir': CONF_DIR,
                                  'random_node': True,
                                  'data_rm_ratio': DATA_RM,
                                  'cat_head': CAT_HEAD,
                                  'con_head': CON_HEAD,
                                  'sample_rate': SAMPLE_RATE,
                                  'window_size': WINDOW_SIZE,
                                  'start_date': Training_start_date,
                                  'small_threshold': 3.0,
                                  'big_threshold': 40.0,
                                  'min_nb': 5,
                                  'test_start_date': Val_start_date,
                                  'test_end_date': Val_end_date}
                    try:
                        if SERVER_NAME == 'server_kdd':
                            dataset = ds.KDD_Data(**prep_param)
                        else:
                            prep_param['topk'] = 5000
                            dataset = ds.GPS_Data(**prep_param)

                        dict_normal, train_data_dict, validate_data_dict = \
                            dataset.prepare_est_pred_with_date(
                                method=TARGET,
                                window=WINDOW_SIZE,
                                mode=MODE,
                                hist_range=HIST_RANGE,
                                least=True,
                                least_threshold=0.5)
                    except KeyboardInterrupt:
                        print("Ctrl-C is pressed, quiting...")
                        sys.exit(0)
