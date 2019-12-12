# Import statements
import numpy as np
import pandas as pd
import tensorflow as tf

class Dataset(object):
    '''
    This is a class that generates batches of training and testing data to be
    used during the training process.
    '''
    def __init__(self, labels, batch_size, shuffle=False):
        self.X = labels['eDWID'].as_matrix()
        self.y = labels['dflag'].as_matrix()
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
          np.random.shuffle(idxs)
        return iter((self.X[idxs[i:i+B]], self.y[idxs[i:i+B]]) for i in range(0, N, B))


def generate_data(patients, patient_labels, batch_size, max_seq_length,
                  num_feats, feats, drop_cols, labels):
    '''
    This method generates batches of patient lables and ID's into 3-D tensors
    appropriate for the RNN.
    '''
    X = np.zeros((batch_size, max_seq_length, num_feats))
    Y = np.zeros((batch_size))
    mask = np.zeros((batch_size, max_seq_length))

    for i, patient in enumerate(patients):
        patient_feats = feats[feats.eDWID == patient].drop(drop_cols, axis=1)
        patient_feats = patient_feats.as_matrix()
        seq_length = patient_feats.shape[0]
        X[i, :seq_length, :] = patient_feats
        mask[i, :seq_length] = True
        mask[i, seq_length:] = False
        Y[i] = labels[labels.eDWID == patient]['dflag'].as_matrix()[0]

    return X, Y, mask


def check_accuracy(sess, dset, x, scores, seq_mask,
                   batch_size, max_seq_length, num_feats, feats, drop_cols,
                   labels, is_training=None):
    '''
    This method computes and displays performance metrics of the network being
    trained and tested.  Accuracy, precision, and true negative rate are all
    calculated.
    '''
    num_correct, num_samples = 0, 0
    num_pos, num_pos_correct = 0, 0
    num_neg, num_neg_correct = 0, 0

    for patients, patient_labels in dset:
      x_batch, y_batch, mask_batch = generate_data(patients, patient_labels,
                                                   batch_size, max_seq_length,
                                                   num_feats, feats, drop_cols,
                                                   labels)
      feed_dict = {x: x_batch, is_training: 0, seq_mask: mask_batch}

      scores_np = sess.run(scores, feed_dict=feed_dict)
      y_pred = scores_np.argmax(axis=1)

      num_samples += x_batch.shape[0]
      num_correct += (y_pred == y_batch).sum()

      num_pos += y_batch.sum()
      num_pos_correct += ((y_pred == y_batch) & (y_batch == 1)).sum()

      num_neg += y_batch.shape[0] - y_batch.sum()
      num_neg_correct += ((y_pred == y_batch) & (y_batch == 0)).sum()

    acc = float(num_correct) / num_samples
    prec = float(num_pos_correct) / num_pos
    true_neg_rate = float(num_neg_correct) / num_neg

    print('Accuracy: Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    print('Precision: Got {} / {} correct ({}%)'.format(num_pos_correct, num_pos, 100*prec))
    print('True Negative Rate: Got {} / {} correct ({}%)'.format(num_neg_correct, num_neg, 100*true_neg_rate))

    return (acc + prec + true_neg_rate) / 3


def train(model_init_fn, optimizer_init_fn, max_seq_length, num_feats,
          patient_batches_train, patient_batches_test, batch_size, feats,
          drop_cols, labels, print_every, num_epochs=1):
    '''
    This method defines the training procedure used to train all RNN's.
    '''
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None,  max_seq_length, num_feats])
    y = tf.placeholder(tf.int32, [None])
    seq_mask = tf.placeholder(tf.bool, [None, max_seq_length])
    is_training = tf.placeholder(tf.bool, name='is_training')

    scores = model_init_fn(x, is_training, seq_mask)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(loss)

    optimizer = optimizer_init_fn()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    with sess:
      sess.run(tf.global_variables_initializer())
      t = 0
      loss_history = []
      perf_history = []
      for epoch in range(num_epochs):
          print('Starting epoch %d' % epoch)
          for (patients, patient_labels) in patient_batches_train:
            x_np, y_np, mask_np = generate_data(patients, patient_labels,
                                                batch_size, max_seq_length,
                                                num_feats, feats, drop_cols,
                                                labels)
            feed_dict = {x: x_np, y: y_np, is_training: 1, seq_mask: mask_np}
            score_np, loss_np, _ = sess.run([scores, loss, train_op], feed_dict=feed_dict)
            loss_history.append(loss_np)
            if t % print_every == 0:
              print('Iteration %d, loss = %.4f' % (t, loss_np))
              perf = check_accuracy(sess, patient_batches_test, x, scores, seq_mask,
                                    batch_size, max_seq_length, num_feats, feats,
                                    drop_cols, labels, is_training=is_training)
              perf_history.append(perf)
              print()
            t += 1
    return loss_history, perf_history
