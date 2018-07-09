import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil
from collections import defaultdict


# NFEATURES = 28**2
# NCLASSES = 10


# Common methods for all models


class base_model(object):
    def __init__(self):
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.label_to_cat = {}
        for key in self.seg_classes.keys():
            for label in self.seg_classes[key]:
                self.label_to_cat[label] = key
        self.regularizers = []

    # High-level interface which runs the constructed computational graph.

    def predict(self, data, cat, labels, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty((size, data.shape[1]))
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], 6))
            batch_cat = np.zeros((self.batch_size))
            # batch_lap = np.zeros((self.batch_size, data.shape[1], data.shape[1]))
            tmp_data = data[begin:end, :]
            tmp_cat = cat[begin:end]
            # tmp_lap = lap[begin:end, :, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            batch_cat[:end - begin] = tmp_cat
            # batch_lap[:end - begin] = tmp_lap
            feed_dict = {self.ph_data: batch_data, self.ph_cat: batch_cat, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, data.shape[1]))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, cat, labels, sess=None):
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

        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, cat, labels, sess)
        # print(predictions)
        ncorrects = np.mean(predictions == labels, axis=1)
        print(ncorrects)
        accuracy = np.mean(ncorrects)
        f1 = 0

        cat_iou = defaultdict(list)

        tot_iou = []
        for i in range(predictions.shape[0]):
            segp = predictions[i, :]
            segl = labels[i, :]
            cat = self.label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]

            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - self.seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - self.seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            cat_iou[cat].append(np.mean(part_ious))
            tot_iou.append(np.mean(part_ious))

        for key, value in cat_iou.items():
            print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
        # print(tot_iou)
        # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.4f} ({:d} / {:d}), iou (weighted): {:.4f}, loss: {:.2e}'.format(
            accuracy, np.sum(predictions == labels), labels.shape[0] * labels.shape[1], np.mean(tot_iou), loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_data, train_cat, train_labels, val_data, val_cat, val_labels, is_continue=True):
        f = open("train.log", "a")
        t_process, t_wall = time.process_time(), time.time()
        if is_continue is True:
            sess = self._get_session(sess=None)
        else:
            sess = tf.Session(graph=self.graph)
            sess.run(self.op_init)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))

        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model')

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))

            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_cat, batch_labels = train_data[idx, :], train_cat[idx], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_cat: batch_cat, self.ph_labels: batch_labels,
                         self.ph_dropout: self.dropout}
            learning_rate, loss_average, loss = sess.run([self.op_train, self.op_loss_average, self.op_loss], feed_dict)
            # print(sess.run([self.op_logits],feed_dict))
            print(loss, loss_average)
            f.write(str(loss) + str(loss_average))
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                f.write('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                f.write('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_cat, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                f.write('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))
                f.write('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    # Methods to construct the computational graph.

    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                # self.pj_graph = tf.placeholder(tf.float32, (self.batch_size, M_0, M_0), 'lapacian')
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, 6), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size, M_0), 'labels')
                self.ph_cat = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_cat, self.ph_dropout)

            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(self.graph, run_meta=run_meta, cmd='op', options=opts)
            print('Total flops' + str(flops.total_float_ops))

            self.op_logits = op_logits
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def inference(self, data, cat, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size 2 x N x M
        first part:
            N: number of signals (samples)
            M: number of vertices (features)
        second part:
            K: k layers of graph
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, cat, dropout)
        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=2)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, regularization

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.8):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class rgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, vertice, F, K, M, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name=''):
        super().__init__()

        # Verify the consistency w.r.t. the number of layers.
        assert len(F) == len(K)

        # Keep the useful Laplacians only. May be zero.
        M_0 = vertice
        # Print information about NN architecture.
        Ngconv = len(F)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(vertice))
        for i in range(Ngconv):
            print('  layer {0}: gconv{0}'.format(i + 1))
            print('    representation: M_{0} * F_{1}= {2} * {3} = {4}'.format(
                i, i + 1, vertice, F[i], vertice * F[i]))
            F_last = F[i - 1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
            print('    biases: F_{} = {}'.format(i + 1, F[i]))
        for i in range(Nfc):
            name = 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
            M_last = M[i - 1] if i > 0 else vertice if Ngconv == 0 else vertice * F[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
            print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Store attributes and bind operations.
        self.F, self.K, self.M = F, K, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)

        # Build the computational graph.
        self.build_graph(M_0)

    def chebyshev5(self, x, L, Fout, K):
        # If K == 1 it is equivalent to fc layer
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = x  # N x M x Fin
        x = tf.expand_dims(x0, 0)

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if K > 1:
            x1 = tf.matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2
        # K x N x M x Fin
        x = tf.transpose(x, perm=[1, 2, 3, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def pairwise_distance(self, point_cloud):
        """Compute pairwise distance of a point cloud.

        Args:
            point_cloud: tensor (batch_size, num_points, num_dims)

        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """

        og_batch_size = point_cloud.get_shape().as_list()[0]
        point_cloud = tf.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = tf.expand_dims(point_cloud, 0)

        point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
        point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
        point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        adj_matrix = tf.exp(-adj_matrix)
        return adj_matrix

    def knn(self, adj_matrix, k=30):
        """Get KNN based on the pairwise distance.
        Args:
            pairwise distance: (batch_size, num_points, num_points)
            k: int

        Returns:
            nearest neighbors: (batch_size, num_points, k)
        """
        # neg_adj = -adj_matrix
        _, nn_idx = tf.nn.top_k(adj_matrix, k=k)
        return nn_idx

    def get_laplacian(self, adj_matrix, normalize=True):
        """Compute pairwise distance of a point cloud.

        Args:
            pairwise distance: tensor (batch_size, num_points, num_points)

        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        if normalize:
            D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
            eye = tf.ones_like(D)
            eye = tf.matrix_diag(eye)
            D = 1 / tf.sqrt(D)
            D = tf.matrix_diag(D)
            L = eye - tf.matmul(tf.matmul(D, adj_matrix), D)
        else:
            D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
            # eye = tf.ones_like(D)
            # eye = tf.matrix_diag(eye)
            # D = 1 / tf.sqrt(D)
            D = tf.matrix_diag(D)
            L = D - adj_matrix
        return L

    def get_one_matrix_knn(self, matrix, k):
        values, indices = tf.nn.top_k(matrix, k,
                                      sorted=False)  # indices will be [[0, 1], [1, 2]], values will be [[6., 2.], [4., 5.]]

        my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_substract = tf.sparse_to_dense(full_indices, matrix.get_shape(), tf.reshape(values, [-1]), default_value=0.,
                                          validate_indices=False)

        # res = matrix - to_substract  # res should be all 0.
        return to_substract

    def _inference(self, x, cat, dropout):
        L = self.pairwise_distance(x)
        # L_ =self.get_laplacian(L,normalize=False)
        # L = tf.stack([self.get_one_matrix_knn(matrix = L[o],k = 30) for o in range(L.get_shape()[0])])
        L = self.get_laplacian(L)
        cat = tf.expand_dims(cat, axis=1)
        cat = tf.one_hot(cat, 16, axis=-1)
        cat = tf.tile(cat, [1, 2048, 1])
        x = tf.concat([x, cat], axis=2)

        x1 = 0 #cache for layer1
        for i in range(len(self.F)):
            with tf.variable_scope('conv{}'.format(i)):
                with tf.name_scope('filter'):
                    if i == 4:
                        x = tf.concat([x, x1], axis=2)
                    x = self.filter(x, L, self.F[i], self.K[i])
                    if i == 1:
                        x1 = x
                    self.regularizers.append(tf.nn.l2_loss(tf.matmul(tf.matmul(tf.transpose(x, perm=[0, 2, 1]), L), x)))
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
        return x
