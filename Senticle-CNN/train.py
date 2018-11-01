import os
import time
import datetime
import tensorflow as tf
import numpy as np
import cnn_tool as tool
from tensorflow import flags
from main import TextCNN
from tensorflow.contrib import learn


def train():
    # # data loading
    data_path = 'preprocessed_SKhynix.csv' # csv 파일로 불러오기
    # 포스코 모델
    # data_path = 'repro_45.csv' # csv 파일로 불러오기
    contents, points = tool.loading_rdata(data_path) # CSV 읽어오기
    vocab_list = tool.cut(contents) # contents 에 모든 기사들을 1개의 리스트에 통합

    # transform document to vector
    max_document_length = 1400
    x, vocabulary, vocab_size = tool.make_vocab(vocab_list, max_document_length)

    tool.save_vocab('news_vocab_sk.txt', contents, max_document_length)
    # tool.save_vocab('news_vocab_posco.txt', vocabulary, max_document_length)

    # vocab = tool.load_vocab('news_vocab_sk.txt')

    print('사전단어수 : %s' %(vocab_size))


    y = tool.make_output(points, threshold = 0)

    # divide dataset into train/test set
    x_train, x_test, y_train, y_test = tool.divide(x, y, train_prop = 0.9)


    # Model Hyperparameters
    flags.DEFINE_integer('embedding_dim', 128, "Dimensionality of embedded vector (default: 128)")
    flags.DEFINE_string('filter_sizes', '3,4,5', "Comma-separated filter sizes (default: '3,4,5')")
    flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size (default: 128)")
    flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")
    flags.DEFINE_float('l2_reg_lambda', 0.1,  "L2 regularization lambda (default: 0.0)")

    # Training parameters
    flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


    FLAGS = flags.FLAGS

    # print('\nParameters : ')
    # for attr, value in sorted(FLAGS.flag_values_dict()):
    #     print('{}={}'.format(attr.upper(), value))
    # print('')

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(sequence_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocab_size=vocab_size,
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            # cnn = CharCNN()

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # initW = tool.load_embedding_vectors(vocabulary)

            # sess.run(cnn.W.assign(initW))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            def batch_iter(data, batch_size, num_epochs, shuffle=True):
                """
                Generates a batch iterator for a dataset.
                """
                data = np.array(data)
                data_size = len(data)
                num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
                for epoch in range(num_epochs):
                    # Shuffle the data at each epoch
                    if shuffle:
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                    else:
                        shuffled_data = data
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size)
                        yield shuffled_data[start_index:end_index]


            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            testpoint = 0
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    if testpoint + 100 < len(x_test):
                        testpoint += 100
                    else:
                        testpoint = 0
                    print("\nEvaluation:")
                    dev_step(x_test[testpoint:testpoint+100], y_test[testpoint:testpoint+100], writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    train()