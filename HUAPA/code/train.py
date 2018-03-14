#-*- coding: utf-8 -*-
#author: Zhen Wu

import os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model import HUAPA


# Data loading params
tf.flags.DEFINE_integer("n_class", 5, "Numbers of class")
tf.flags.DEFINE_string("dataset", 'yelp13', "The dataset")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("hidden_size", 100, "hidden_size of rnn")
tf.flags.DEFINE_integer('max_sen_len', 50, 'max number of tokens per sentence')
tf.flags.DEFINE_integer('max_doc_len', 40, 'max number of tokens per sentence')
tf.flags.DEFINE_float("lr", 0.005, "Learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")
trainset = Dataset('../../data/'+FLAGS.dataset+'/train.ss')
devset = Dataset('../../data/'+FLAGS.dataset+'/dev.ss')
testset = Dataset('../../data/'+FLAGS.dataset+'/test.ss')

alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)
embeddingpath = '../../data/'+FLAGS.dataset+'/embedding.txt'
embeddingfile, wordsdict = data_helpers.load_embedding(embeddingpath, alldata, FLAGS.embedding_dim)
del alldata
print("Loading data finished...")

usrdict, prddict = trainset.get_usr_prd_dict()
trainbatches = trainset.batch_iter(usrdict, prddict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
                                 FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)
devset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
testset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)


with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        huapa = HUAPA(
            max_sen_len = FLAGS.max_sen_len,
            max_doc_len = FLAGS.max_doc_len,
            class_num = FLAGS.n_class,
            embedding_file = embeddingfile,
            embedding_dim = FLAGS.embedding_dim,
            hidden_size = FLAGS.hidden_size,
            user_num = len(usrdict),
            product_num = len(prddict)
        )
        huapa.build_model()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(huapa.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Save dict
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("../checkpoints/"+FLAGS.dataset+"/"+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with open(checkpoint_dir + "/wordsdict.txt", 'wb') as f:
            pickle.dump(wordsdict, f)
        with open(checkpoint_dir + "/usrdict.txt", 'wb') as f:
            pickle.dump(usrdict, f)
        with open(checkpoint_dir + "/prddict.txt", 'wb') as f:
            pickle.dump(prddict, f)

        sess.run(tf.global_variables_initializer())

        def train_step(batch):
            u, p, x, y, sen_len, doc_len = zip(*batch)
            feed_dict = {
                huapa.userid: u,
                huapa.productid: p,
                huapa.input_x: x,
                huapa.input_y: y,
                huapa.sen_len: sen_len,
                huapa.doc_len: doc_len
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, huapa.loss, huapa.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def predict_step(u, p, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                huapa.userid: u,
                huapa.productid: p,
                huapa.input_x: x,
                huapa.input_y: y,
                huapa.sen_len: sen_len,
                huapa.doc_len: doc_len
            }
            step, loss, accuracy, correct_num, mse = sess.run(
                [global_step, huapa.loss, huapa.accuracy, huapa.correct_num, huapa.mse],
                feed_dict)
            return correct_num, accuracy, mse

        def predict(dataset, name=None):
            acc = 0
            rmse = 0.
            for i in xrange(dataset.epoch):
                correct_num, _, mse = predict_step(dataset.usr[i], dataset.prd[i], dataset.docs[i],
                                                   dataset.label[i], dataset.sen_len[i], dataset.doc_len[i], name)
                acc += correct_num
                rmse += mse
            acc = acc * 1.0 / dataset.data_size
            rmse = np.sqrt(rmse / dataset.data_size)
            return acc, rmse

        topacc = 0.
        toprmse = 0.
        better_dev_acc = 0.
        predict_round = 0

        # Training loop. For each batch...
        for tr_batch in trainbatches:
            train_step(tr_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                predict_round += 1
                print("\nEvaluation round %d:" % (predict_round))

                dev_acc, dev_rmse = predict(devset, name="dev")
                print("dev_acc: %.4f    dev_RMSE: %.4f" % (dev_acc, dev_rmse))
                # test_acc, test_rmse = predict(testset, name="test")
                # print("test_acc: %.4f    test_RMSE: %.4f" % (test_acc, test_rmse))

                # print topacc with best dev acc
                if dev_acc >= better_dev_acc:
                    better_dev_acc = dev_acc
                    topacc = test_acc
                    toprmse = test_rmse
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                print("topacc: %.4f   RMSE: %.4f" % (topacc, toprmse))
