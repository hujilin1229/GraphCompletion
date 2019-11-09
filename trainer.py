import numpy as np
import tensorflow as tf
from tqdm import trange
import scipy
import time
import pandas as pd

import graph
from model_gcnn import Model
from utils import BatchLoader, evaluate_result, LSM_Loader, softmax, fill_mean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor

"""
Trainer: 

1. Initializes model
2. Train
3. Test
"""


class Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.checkpoint_secs = config.checkpoint_secs
        self.log_step = config.log_step
        self.num_epoch = config.num_epochs
        self.stop_win_size = config.stop_win_size
        self.stop_early = config.stop_early

        ## import data Loader ##ir
        batch_size = config.batch_size
        server_name = config.server_name
        mode = config.mode
        target = config.target
        sample_rate = config.sample_rate
        win_size = config.win_size
        hist_range = config.hist_range
        s_month = config.s_month
        e_month = config.e_month
        e_date = config.e_date
        s_date = config.s_date
        data_rm = config.data_rm
        coarsening_level = config.coarsening_level
        cnn_mode = config.conv
        is_coarsen = config.is_coarsen

        self.data_loader = BatchLoader(server_name, mode, target, sample_rate, win_size,
                                       hist_range, s_month, s_date, e_month, e_date,
                                       data_rm, batch_size, coarsening_level, cnn_mode,
                                       is_coarsen)

        actual_node = self.data_loader.adj.shape[0]
        if config.conv == 'gcnn':
            graphs = self.data_loader.graphs
            if config.is_coarsen:
                L = [graph.laplacian(A, normalized=config.normalized) for A in graphs]
            else:
                L = [graph.laplacian(self.data_loader.adj,
                                      normalized=config.normalized)] * len(graphs)
        elif config.conv == 'cnn':
            L = [actual_node]
            tmp_node = actual_node
            while tmp_node > 0:
                tmp_node = int(tmp_node / 2)
                L.append(tmp_node)
        else:
            raise ValueError(
                "Unsupported config.conv {}".format(
                    config.conv))

        tf.reset_default_graph()
        ## define model ##
        self.model = Model(config, L, actual_node)

        ## model saver / summary writer ##
        self.saver = tf.train.Saver()
        self.model_saver = tf.train.Saver(self.model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        # Checkpoint
        # meta file: describes the saved graph structure, includes
        # GraphDef, SaverDef, and so on; then apply
        # tf.train.import_meta_graph('/tmp/model.ckpt.meta'),
        # will restore Saver and Graph.

        # index file: it is a string-string immutable
        # table(tensorflow::table::Table). Each key is a name of a tensor
        # and its value is a serialized BundleEntryProto.
        # Each BundleEntryProto describes the metadata of a
        # tensor: which of the "data" files contains the content of a tensor,
        # the offset into that file, checksum, some auxiliary data, etc.
        #
        # data file: it is TensorBundle collection, save the values of all variables.

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        #
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        # init = tf.global_variables_initializer()
        # self.sess = tf.Session(config=sess_config)
        # self.sess.run(init)

    def train(self, val_best_score=10, save=False, index=1, best_model=None):
        print("[*] Checking if previous run exists in {}"
              "".format(self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if tf.train.latest_checkpoint(self.model_dir) is not None:
            print("[*] Saved result exists! loading...")
            self.saver.restore(
                self.sess,
                latest_checkpoint
            )
            print("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            print("[*] No previous result")
            self.b_pretrain_loaded = False

        print("[*] Training starts...")
        self.model_summary_writer = None

        val_loss = 0
        lr = 0
        tmp_best_loss = float('+inf')
        validation_loss_window = np.zeros(self.stop_win_size)
        validation_loss_window[:] = float('+inf')
        ##Training
        for n_epoch in trange(self.num_epoch, desc="Training[epoch]"):
            self.data_loader.reset_batch_pointer(0)
            loss_epoch = []
            for k in trange(self.data_loader.sizes[0], desc="[per_batch]"):
                # Fetch training data
                batch_x, batch_y, weight_y,\
                count_y, _ = self.data_loader.next_batch(0)

                feed_dict = {
                    self.model.cnn_input: batch_x,
                    self.model.output_label: batch_y,
                    self.model.ph_labels_weight: weight_y,
                    self.model.is_training: True
                }
                res = self.model.train(self.sess, feed_dict, self.model_summary_writer,
                                       with_output=True)
                loss_epoch.append(res['loss'])
                lr = res['lr']
                self.model_summary_writer = self._get_summary_writer(res)

            val_loss = self.validate()
            train_loss = np.mean(loss_epoch)

            validation_loss_window[n_epoch % self.stop_win_size] = val_loss

            if self.stop_early:
                if np.abs(validation_loss_window.mean() - val_loss) < 1e-4:
                    print('Validation loss did not decrease. Stopping early.')
                    break

            if n_epoch % 10 == 0:
                if save:
                    self.saver.save(self.sess, self.model_dir)
                if val_loss < val_best_score:
                    val_best_score = val_loss
                    best_model = self.model_dir
                if val_loss < tmp_best_loss:
                    tmp_best_loss = val_loss
                print("Searching {}...".format(index))
                print("Epoch {}: ".format(n_epoch))
                print("LR: ", lr)
                print("  Train Loss: ", train_loss)
                print("  Validate Loss: ", val_loss)
                print("  Current Best Loss: ", val_best_score)
                print("  Current Model Dir: ", best_model)

        return tmp_best_loss

    def validate(self):

        loss = []
        for n_sample in trange(self.data_loader.sizes[1], desc="Validating"):
            batch_x, batch_y, weight_y, count_y,\
            _ = self.data_loader.next_batch(1)

            feed_dict = {
                self.model.cnn_input: batch_x,
                self.model.output_label: batch_y,
                self.model.ph_labels_weight: weight_y,
                self.model.is_training: False
            }
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            loss.append(res['loss'])

        return np.nanmean(loss)

    def test(self):

        loss = []
        gt_y = []
        pred_y = []
        w_y = []
        counts_y = []
        vel_list_y = []
        for n_sample in trange(self.data_loader.sizes[2], desc="Testing"):
            batch_x, batch_y, weight_y, \
            count_y, vel_list = self.data_loader.next_batch(2)

            feed_dict = {
                self.model.cnn_input: batch_x,
                self.model.output_label: batch_y,
                self.model.ph_labels_weight: weight_y,
                self.model.is_training: False
            }
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            loss.append(res['loss'])
            gt_y.append(batch_y)
            w_y.append(weight_y)
            counts_y.append(count_y)
            vel_list_y.append(vel_list)
            pred_y.append(res['pred'])

        final_gt = np.concatenate(gt_y, axis=0)
        final_pred = np.concatenate(pred_y, axis=0)
        final_weight = np.concatenate(w_y, axis=0)
        final_count = np.concatenate(counts_y, axis=0)
        final_vel_list = np.concatenate(vel_list_y, axis=0)

        result_dict = {'ground_truth': final_gt,
                       'prediction': final_pred,
                       'weight': final_weight,
                       'count': final_count,
                       'vel_list': final_vel_list}

        test_loss = np.mean(loss)
        print("Test Loss: ", test_loss)

        return result_dict
            # self.model_summary_writer = self._get_summary_writer(res)

    def _get_summary_writer(self, result):
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None

class Regression_Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        server_name = config.server_name
        mode = config.mode
        target = config.target
        sample_rate = config.sample_rate
        win_size = config.win_size
        hist_range = config.hist_range
        s_month = config.s_month
        e_month = config.e_month
        e_date = config.e_date
        s_date = config.s_date
        data_rm = config.data_rm
        coarsening_level = config.coarsening_level
        cnn_mode = config.conv
        is_coarsen = config.is_coarsen
        batch_size = -1
        self.filter = config.filter
        self.pool = config.pool
        self.p_size = config.p_size

        self.data_loader = BatchLoader(server_name, mode, target, sample_rate, win_size,
                                       hist_range, s_month, s_date, e_month, e_date,
                                       data_rm, batch_size, coarsening_level, cnn_mode,
                                       is_coarsen)

    def mean_gt(self, batch_y):

        mean_y = np.mean(batch_y, axis=0)
        mean_y = self.softmax(mean_y, n_axis=-1, exp=False)

        return mean_y

    def fill_mean(self, source, mean, zero_fill=True):

        num_record = source.shape
        tile_shape = [1] * len(num_record)
        tile_shape[0] = num_record[0]
        tile_shape = tuple(tile_shape)
        tile_mean = np.tile(mean, tile_shape)

        sum_source = np.sum(source, axis=-1)
        if zero_fill:
            selected_pos = sum_source < 0.01
        else:
            selected_pos = sum_source > 0.9
        source[selected_pos] = tile_mean[selected_pos]

        return source

    def train(self, val_best_score=10):
        print("[*] Now in {} regression process...".format(self.filter))
        batch_x, batch_y, weight_y, count_y, _ = self.data_loader.next_batch(0)
        num_bins = batch_x.shape[-1]
        self.mean_y = self.mean_gt(batch_y)
        self.mean_y[np.isnan(self.mean_y)] = 0.0

        batch_y = self.fill_mean(batch_y, self.mean_y)

        if self.pool:
            batch_x = self.cluster_edges(batch_x, self.p_size)
        else:
            batch_x = self.fill_mean(batch_x, self.mean_y)
        # Instanciate a Gaussian Process model

        print("Number of Bin is ", num_bins)
        self.gps = []
        for i in range(num_bins):
            if self.filter == 'gpr':
                # kernel = C(1.0, (1e-2, 4e-1)) * RBF(1.0, (1e-1, 1.0))
                gp = GaussianProcessRegressor()
            elif self.filter == 'rf':
                nb_estimators = 50
                max_features = 'log2'
                gp = RandomForestRegressor(n_estimators=nb_estimators,
                                           max_features=max_features)
            else:
                print("Please specify a correct regression model...")
            gp.fit(batch_x[..., i], batch_y[..., i])
            self.gps.append(gp)

        val_y = self.test()

        # KL Divergence
        val_kl = evaluate_result(val_y, method='KL')
        val_ks = evaluate_result(val_y, method='KS-test')
        val_wasser = evaluate_result(val_y, method='wasser')
        print("     KL   |   KS  |   Wasser  |")
        print("     {}   |   {}  |     {}   |".format(
            val_kl, val_ks, val_wasser))

    def cluster_edges(self, data_array, p_size=8):

        size_data_array = data_array.shape[1]
        output_len = int(size_data_array / p_size)
        output_array = np.zeros((data_array.shape[0], output_len , data_array.shape[2]))
        for i in range(output_len):
            output_array[:, i, :] = np.max(data_array[:, i*p_size:(i+1)*p_size, :], axis=1)

        return output_array

    def softmax(self, x, n_axis=-1, exp=True):

        if exp:
            # take the sum along the specified axis
            x = np.exp(x)
        else:
            # in case there's negative value in the output
            x_min = np.expand_dims(np.min(x, axis=n_axis), n_axis)
            x = (x - x_min)
            # x[x<0] = 0.

        ax_sum = np.expand_dims(np.sum(x, axis=n_axis), n_axis)

        return x / ax_sum

    def test(self, val=True):

        if val:
            val_ind = 1
        else:
            val_ind = 2

        batch_x, batch_y, weight_y, count_y, vel_list = self.data_loader.next_batch(val_ind)
        if self.pool:
            batch_x = self.cluster_edges(batch_x, self.p_size)
        else:
            batch_x = self.fill_mean(batch_x, self.mean_y)
        num_bins = batch_x.shape[-1]
        output_y = []
        for i in range(num_bins):
            batch_x_i = batch_x[..., i]
            val_y_i = self.gps[i].predict(batch_x_i)
            output_y.append(val_y_i[..., np.newaxis])
        output_y = np.concatenate(output_y, axis=-1)
        output_y = self.softmax(output_y, -1, exp=False)

        val_result = {'ground_truth': batch_y,
                      'prediction': output_y,
                      'weight': weight_y,
                      'count': count_y,
                      'vel_list': vel_list}

        return val_result

    def mean_test(self, val=True):

        if val:
            val_ind = 1
        else:
            val_ind = 2

        batch_x, batch_y, weight_y, count_y, vel_list = self.data_loader.next_batch(val_ind)
        batch_y_copy = batch_y.copy()
        output_y = self.fill_mean(batch_y_copy, self.mean_y, zero_fill=False)

        val_result = {'ground_truth': batch_y,
                      'prediction': output_y,
                      'weight': weight_y,
                      'count': count_y,
                      'vel_list': vel_list}

        return val_result

class LSM_Trainer():
    """
    This class is an implementation of Latent Space Model for Road Network

    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        self.converge_loss = config.converge_loss
        self.T = config.T
        self.k = config.k
        self.lamda = config.lamda
        self.gamma = config.gamma
        self.eval_frequency = config.eval_frequency
        self.epochs = config.num_epochs

        server_name = config.server_name
        mode = config.mode
        target = config.target
        sample_rate = config.sample_rate
        win_size = config.win_size
        hist_range = config.hist_range
        s_month = config.s_month
        e_month = config.e_month
        e_date = config.e_date
        s_date = config.s_date
        data_rm = config.data_rm

        self.data_loader = LSM_Loader(server_name, mode, target, sample_rate, win_size,
                                       hist_range, s_month, s_date, e_month, e_date,
                                       data_rm)

        self.nbins = len(hist_range) - 1
        self.L = self.data_loader.Lap
        self.W = self.data_loader.W
        self.D = self.data_loader.D
        self.nb_nodes = self.L.shape[0]
        self.row_ind = self.data_loader.row_ind
        self.col_ind = self.data_loader.col_ind

    def train(self):

        data_dict = self.data_loader.val_data_dict
        # The following data has three layers: 1. num_record, 2. num_dim, 3. sp.matrix
        Gts = data_dict['Gt']
        Vts = data_dict['Vt']
        Yts = data_dict['Yt']
        Wts = data_dict['Wt']
        loss_list = []
        for ind, Gt in enumerate(Gts):
            train_weight = Vts[ind][0]
            val_weight = Wts[ind][0]
            for b_i in range(self.nbins):
                train_data = Gt[b_i]
                val_data = Yts[ind][b_i]
                loss_i, _ = self.fit(train_data, train_weight, val_data, val_weight)
                loss_list.append(loss_i)

        print("The average loss in training is ", np.nanmean(loss_list))

    def test(self):

        data_dict = self.data_loader.test_data_dict
        # The following data has three layers: 1. num_record, 2. num_dim, 3. sp.matrix
        Gts = data_dict['Gt']
        Vts = data_dict['Vt']
        Yts = data_dict['Yt']
        Wts = data_dict['Wt']
        Cts = data_dict['Ct']
        Vel_list_ts = data_dict['Velt']

        list_gt = []
        list_pred = []
        list_cnt = []
        list_wt = []
        list_vel_list = []
        for ind, Gt in enumerate(Gts):
            train_weight = Vts[ind][0]
            val_weight = Wts[ind][0]
            com_complete = []
            for b_i in range(self.nbins):
                train_data = Gt[b_i]
                val_data = Yts[ind][b_i]
                loss_i, complete_i = self.fit(train_data, train_weight, val_data, val_weight)
                com_complete.append(complete_i)
            count = Cts[ind][0].toarray()
            vel_list_array = Vel_list_ts[ind][0]
            val_weight = val_weight.toarray()
            if np.sum(val_weight) == 0:
                continue
            gt, pred, cnt, wt, v_list = self.cvt2edge_array(
                Gt, com_complete, val_weight, vel_list_array, count)

            list_gt.append(np.expand_dims(gt, axis=0))
            list_pred.append(np.expand_dims(pred, axis=0))
            list_cnt.append(np.expand_dims(cnt, axis=0))
            list_wt.append(np.expand_dims(wt, axis=0))
            list_vel_list.append(np.expand_dims(v_list, axis=0))

        array_gt = np.concatenate(list_gt, axis=0)
        array_pred = np.concatenate(list_pred, axis=0)
        array_cnt = np.concatenate(list_cnt, axis=0)
        array_wt = np.concatenate(list_wt, axis=0)
        array_vel_list = np.concatenate(list_vel_list, axis=0)

        array_pred = softmax(array_pred, -1, exp=True)
        val_result = {'ground_truth': array_gt,
                      'prediction': array_pred,
                      'weight': array_wt,
                      'count': array_cnt,
                      'vel_list': array_vel_list}

        return val_result

    def cvt2edge_array(self, val_data, complete,
                       weight, vel_list, count):

        num_edge = len(self.row_ind)
        true_array = np.zeros((num_edge, self.nbins))
        pred_array = np.zeros((num_edge, self.nbins))
        final_count = np.zeros(num_edge)
        final_vel_list = pd.DataFrame(data={0:[[]]*num_edge})
        out_weight = np.zeros(num_edge, dtype=np.int)

        for i in range(num_edge):
            row_i = self.row_ind[i]
            col_i = self.col_ind[i]
            weight_i = weight[row_i, col_i]
            if weight_i == 1:
                final_vel_list.at[i, 0] = vel_list[row_i, col_i]
                final_count[i] = count[row_i, col_i]
                out_weight[i] = 1
                for bi in range(self.nbins):
                    val_data_i = val_data[bi].todense()
                    true_array[i, bi] = val_data_i[row_i, col_i]
                    pred_array[i, bi] = complete[bi][0][row_i, col_i]

        return true_array, pred_array, final_count, out_weight, final_vel_list.values

    def re_arrange(self, val_data, complete, weight, count):

        select_pos = weight == 1
        num_val = int(np.sum(weight))

        true_array = np.zeros((num_val, self.nbins))
        pred_array = np.zeros((num_val, self.nbins))
        final_count = count[select_pos]
        out_weight = np.ones(num_val)
        for bi in range(self.nbins):
            val_data_i = val_data[bi].todense()
            true_array[:, bi] = val_data_i[select_pos]
            pred_array[:, bi] = complete[bi][0][select_pos]

        return true_array, pred_array, final_count, out_weight

    def re_arrange_vel_list(self, val_data, complete,
                            weight, vel_list, count):

        select_pos = weight == 1
        num_val = int(np.sum(weight))
        true_array = np.zeros((num_val, self.nbins))
        pred_array = np.zeros((num_val, self.nbins))
        final_count = count[select_pos]
        final_vel_list = vel_list[select_pos]
        out_weight = np.ones(num_val)
        for bi in range(self.nbins):
            val_data_i = val_data[bi].todense()
            true_array[:, bi] = val_data_i[select_pos]
            pred_array[:, bi] = complete[bi][0][select_pos]

        return true_array, pred_array, final_count, out_weight, final_vel_list


    def weighted_mape_tf(self, y_true, y_pred, weight):

        y_true_f = y_true[weight == 1]
        y_pred_f = y_pred[weight == 1]

        wmape = np.mean(np.abs((y_true_f - y_pred_f) / y_true_f)) * 100

        return wmape

    def weighted_l2(self, y_true, y_pred, weight):

        y_true_f = y_true[weight == 1]
        y_pred_f = y_pred[weight == 1]

        if len(y_true_f) == 0:
            return 0.

        wl2 = np.sum(np.square(y_true_f - y_pred_f)) / np.sum(weight)

        return wl2

    def _ut_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('u_ts', shape, tf.float32, initializer=initial)

        tf.summary.histogram(var.op.name, var)
        return var

    def fit(self, train_data, train_labels, val_data, val_labels):
        """Implement the global learning algorithm for LSM"""

        # Initialize Ut, B and A
        self.U_t = [np.random.rand(self.nb_nodes, self.k)
                    for i in range(self.T)]
        self.B = np.random.rand(self.k, self.k)
        self.A = np.random.rand(self.k, self.k)
        self.G_t = [train_data.todense()]
        self.Y_t = [train_labels.todense()]
        self.W_t = [val_labels.todense()]
        self.V_t = [val_data.todense()]
        self.E = np.identity(self.nb_nodes)

        loss_average, complete = self.global_learning()
        return loss_average, complete

    def global_learning(self):
        t_process, t_wall = time.process_time(), time.time()
        # Iterating.
        losses = []
        error = 100.0
        loss_average = 100
        step = 0
        while (error > self.converge_loss) and step < self.epochs:
            step += 1
            self.update_ut_tr()
            self.update_B()
            self.update_A()
            pre_loss = loss_average
            loss_average = self.loss()
            # print("Step: {}, Loss: {}".format(step, loss_average))
            error = abs(loss_average - pre_loss)
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0:
                # print('step {}:'.format(step))
                # print('  loss_average = {:.2e}'.format(loss_average))
                # print('U: ', self.U_t)
                # print('B: ', self.B)
                string, loss, _ = self.evaluate(self.V_t, self.W_t)
                losses.append(loss)
                # print('  validation {}'.format(string))
                # print('  time: {:.0f}s (wall {:.0f}s)'.format(
                #     time.process_time() - t_process, time.time() - t_wall))

        known_loss, loss, complete = self.evaluate(self.V_t, self.W_t)

        return loss, complete

    def loss(self):
        """Adds to the inference model the layers required to generate loss."""
        first_losses = 0
        for i in range(self.T):
            UtBUtT = np.matmul(self.U_t[i], self.B)
            UtBUtT = np.matmul(UtBUtT, np.transpose(self.U_t[i]))
            Gt_UBUT = self.G_t[i] - UtBUtT
            Yt_GtUBUT = np.multiply(self.Y_t[i], Gt_UBUT)
            first_losses += np.sum(np.power(Yt_GtUBUT, 2))
        loss_second_item = 0
        for i in range(self.T):
            UtL = scipy.sparse.csr_matrix.dot(self.U_t[i].T, self.L)
            UtLUtT = np.matmul(UtL, self.U_t[i])
            loss_second_item += np.trace(UtLUtT) * self.lamda
        loss_third_item = 0
        for i in range(1, self.T):
            UtBUtT = scipy.sparse.csr_matrix.dot(self.U_t[i - 1], self.A)
            GtMinusUtBUtT = np.subtract(self.U_t[i], UtBUtT)
            loss_third_item += np.sum(np.power(GtMinusUtBUtT, 2)) * self.gamma
        regularization = loss_third_item + loss_second_item
        # print("----------------------------")
        # print("first loss: ", first_losses)
        # print("regularization: ", regularization)
        # print("B: ", self.B)
        return first_losses + regularization

    def evaluate(self, batch_data, batch_yval,
                 train_mean=None, small_threshold=0.0,
                 large_threshold=1.0):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """

        losses = []
        complete_list = []
        for i in range(self.T):
            complete_i = np.matmul(self.U_t[i], self.B)
            complete_i = np.matmul(complete_i, np.transpose(self.U_t[i]))
            if train_mean is not None:
                unreasonable_index = np.logical_or(complete_i > large_threshold,
                                                   complete_i < small_threshold)

                complete_i[unreasonable_index] = train_mean[unreasonable_index]
            # # in case the place with weight has no data
            # batch_yval[i][batch_data[i]==0.0] = 0
            # complete_i[batch_yval[i] == 0] = 0.0
            loss_i = self.weighted_l2(
                batch_data[i], complete_i, batch_yval[i])
            complete_list.append(complete_i)
            losses.append(loss_i)
        loss = np.nanmean(losses)
        string = 'loss: {:.2e}'.format(loss)

        return string, loss, complete_list

    def update_ut_tr(self):

        for i in range(self.T):
            u_t_b_T = np.matmul(self.U_t[i], np.transpose(self.B))
            u_t_b = np.matmul(self.U_t[i], self.B)

            if self.T == 1:
                u_t_a = np.zeros([self.nb_nodes, self.k])
            elif i == 0:
                u_t_a = np.matmul(self.U_t[i + 1], np.transpose(self.A))
            elif i == self.T - 1:
                u_t_a = np.matmul(self.U_t[i - 1], self.A)
            else:
                u_t_a = np.matmul(self.U_t[i - 1], self.A) + \
                    np.matmul(self.U_t[i + 1], np.transpose(self.A))

            # calculate lambda * W * U_t
            w_ut = np.matmul(self.W, self.U_t[i]) * self.lamda

            # calcualte Y_t element wise product G
            y_t_G_t = np.multiply(self.Y_t[i], self.G_t[i])
            y_tT_G_tT = np.multiply(np.transpose(
                self.Y_t[i]), np.transpose(self.G_t[i]))

            # combine fractions
            fraction = np.matmul(
                y_t_G_t, u_t_b_T) + np.matmul(y_tT_G_tT, u_t_b) + w_ut + self.gamma * u_t_a

            # calculate Yt Hadamard UtBtUtT
            u_t_b_u_t = np.matmul(u_t_b, np.transpose(self.U_t[i]))
            Yt_u_t_b_u_t = np.multiply(self.Y_t[i], u_t_b_u_t)
            u_t_b_u_t = np.matmul(Yt_u_t_b_u_t, (u_t_b_T + u_t_b))

            # calculate lambda*D*U_t
            d_ut = np.matmul(self.D, self.U_t[i]) * self.lamda
            # d_ut = np.matmul(self.D, self.U_t[i]) * self.lamda

            # calculate U_t + U_t*A*AT
            Ut_At = np.matmul(self.U_t[i], self.A)
            Ut_At = np.matmul(Ut_At, np.transpose(self.A))
            Ut_At += self.U_t[i]
            Ut_At *= self.gamma

            # calculate denominator
            denominator = u_t_b_u_t + d_ut + Ut_At
            # calculate combine and pow
            # com_fraction = np.nan_to_num(fraction / denominator)
            com_fraction = np.divide(fraction, denominator,
                                     out=np.zeros_like(fraction), where=denominator != 0)
            com_fraction = np.power(com_fraction, 1)
            self.U_t[i] = np.multiply(self.U_t[i], com_fraction)
            # print("Ut: ", self.U_t[i])

    def update_ut_kdd(self):

        for i in range(self.T):
            u_t_b = np.matmul(self.U_t[i], np.transpose(self.B)) + \
                np.matmul(self.U_t[i], self.B)
            if self.T == 1:
                u_t_a = np.zeros([self.nb_nodes, self.k])
            elif i == 0:
                u_t_a = np.matmul(self.U_t[i + 1], np.transpose(self.A))
            elif i == self.T - 1:
                u_t_a = np.matmul(self.U_t[i - 1], self.A)
            else:
                u_t_a = np.matmul(self.U_t[i - 1], self.A) + \
                    np.matmul(self.U_t[i + 1], np.transpose(self.A))

            # calculate lambda * W * U_t
            w_ut = np.matmul(self.W, self.U_t[i]) * self.lamda

            # calcualte Y_t element wise product G
            y_t_G = np.multiply(self.Y_t[i], self.G_t[i])

            # combine fractions
            fraction = np.matmul(y_t_G, u_t_b) + w_ut + self.gamma * u_t_a

            # calculate Yt Hadamard UtBtUtT
            u_t_b_u_t = np.matmul(self.U_t[i], self.B)
            u_t_b_u_t = np.matmul(u_t_b_u_t, np.transpose(self.U_t[i]))
            u_t_b_u_t = np.multiply(self.Y_t[i], u_t_b_u_t)
            u_t_b_u_t = np.matmul(u_t_b_u_t, u_t_b)

            # calculate lambda*D*U_t
            d_ut = np.matmul(self.D, self.U_t[i]) * self.lamda

            # calculate U_t + U_t*A*AT
            if self.T == 1:
                Ut_At = 0
            elif i == 0:
                Ut_At = np.matmul(self.U_t[i], self.A)
                Ut_At = np.matmul(Ut_At, np.transpose(self.A)) * self.gamma
            elif i == self.T - 1:
                Ut_At = self.U_t[i] * self.gamma
            else:
                Ut_At = np.matmul(self.U_t[i], self.A)
                Ut_At = np.matmul(Ut_At, np.transpose(self.A))
                Ut_At += self.U_t[i]
                Ut_At *= self.gamma

            # calculate denominator
            denominator = u_t_b_u_t + d_ut + Ut_At
            # calculate combine and pow
            # com_fraction = np.nan_to_num(fraction / denominator)
            com_fraction = np.divide(fraction, denominator,
                                     out=np.zeros_like(fraction),
                                     where=denominator != 0)
            com_fraction = np.power(com_fraction, 0.25)
            self.U_t[i] = np.multiply(self.U_t[i], com_fraction)
            # print("Ut: ", self.U_t[i])

    def update_ut_new(self):

        for i in range(self.T):
            ut_b = np.matmul(self.U_t[i], self.B)
            w_ut = np.matmul(self.W, self.U_t[i])
            g_u_b = scipy.sparse.csr_matrix.dot(self.G_t[i], ut_b)
            fraction = g_u_b + 0.5 * self.lamda * w_ut

            ut_b_uT = np.matmul(ut_b, self.U_t[i].T)
            ut_b_uT_ub = np.matmul(ut_b_uT, ut_b)
            d_u = scipy.sparse.csr_matrix.dot(self.D, self.U_t[i])
            denominator = ut_b_uT_ub + self.lamda * d_u

            com_fraction = np.divide(fraction, denominator,
                                     out=np.zeros_like(fraction),
                                     where=denominator != 0)

            self.U_t[i] = np.multiply(
                self.U_t[i], np.power(com_fraction, 0.25))
            # print("Ut: ")
            # print(self.U_t[i])

    def update_B_new(self):

        fraction = np.zeros([self.k, self.k])
        for i in range(self.T):
            g_u = np.matmul(self.G_t[i], self.U_t[i])
            uT_g_u = np.matmul(self.U_t[i].T, g_u)
            fraction += uT_g_u

        denominator = np.zeros([self.k, self.k])
        for i in range(self.T):
            uT_u = np.matmul(self.U_t[i].T, self.U_t[i])
            uT_u_b = np.matmul(uT_u, self.B)
            uT_u_b_uT_u = np.matmul(uT_u_b, uT_u)
            denominator += uT_u_b_uT_u

        # print("denominator_B:")
        # print(denominator)
        com_fraction = np.divide(fraction, denominator,
                                 out=np.zeros_like(fraction),
                                 where=denominator != 0)
        self.B = np.multiply(self.B, com_fraction)

    def update_B(self):

        fraction = np.zeros([self.k, self.k])
        for i in range(self.T):
            Y_t_G_t = np.multiply(self.Y_t[i], self.G_t[i])
            fraction_i = np.matmul(np.transpose(self.U_t[i]), Y_t_G_t)
            fraction_i = np.matmul(fraction_i, self.U_t[i])
            fraction = np.add(fraction, fraction_i)

        denominator = np.zeros([self.k, self.k])
        for i in range(self.T):
            U_B_UT = np.matmul(self.U_t[i], self.B)
            U_B_UT = np.matmul(U_B_UT, np.transpose(self.U_t[i]))
            Y_t_U_B_UT = np.multiply(self.Y_t[i], U_B_UT)
            fraction_i = np.matmul(np.transpose(self.U_t[i]), Y_t_U_B_UT)
            fraction_i = np.matmul(fraction_i, self.U_t[i])
            denominator = np.add(denominator, fraction_i)
        # print("denominator_B:")
        # print(denominator)
        com_fraction = np.divide(fraction, denominator,
                                 out=np.zeros_like(fraction),
                                 where=denominator != 0)
        self.B = np.multiply(self.B, com_fraction)

    def update_A(self):

        fraction = np.zeros([self.k, self.k])
        for i in range(1, self.T):
            fraction_i = np.matmul(np.transpose(self.U_t[i - 1]), self.U_t[i])
            fraction = np.add(fraction, fraction_i)

        denominator = np.zeros([self.k, self.k])
        for i in range(1, self.T):
            fraction_i = np.matmul(np.transpose(
                self.U_t[i - 1]), self.U_t[i - 1])
            fraction_i = np.matmul(fraction_i, self.A)
            denominator = np.add(denominator, fraction_i)
        com_fraction = np.divide(fraction, denominator,
                                 out=np.zeros_like(fraction),
                                 where=denominator != 0)
        self.A = np.multiply(self.A, com_fraction)
