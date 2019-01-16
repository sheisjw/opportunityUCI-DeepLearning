import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from scipy import stats
import h5py
import os
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size / 2)
        start = int(start)

def segment_opp(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,77))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
            #print("x_start_end",x_train[start:end])
            # segs =  x_train[start:end]
            # segments = np.concatenate((segments,segs))
            # segments = np.vstack((segments,x_train[start:end]))
            # segments = np.vstack([segments,segs])
            # segments = np.vstack([segments,x_train[start:end]])
            # labels = np.append(labels,stats.mode(y_train[start:end]))
    return segments, labels

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test, dataset, input_width):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        print("len(x_train[0])",len(X_train[0]))

        # DEFINING THE MODEL
        if dataset=="opp":
            print("opp")
            self.input_height = 1
            self.input_width = input_width #or 90 for actitracker
            self.num_labels = 18  #or 6 for actitracker
            self.num_channels = 77 #or 3 for actitracker
        else:
            print("wrong dataset")


        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 10
        self.batch_size = 64

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        print("n_inputs len(X_train[0][0])",len(X_train[0][0]))
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = self.num_labels  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, config.n_inputs])

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)

    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    # lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    # lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,lstm_cell_3], state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


print("starting...")
start_time = time.time()

# DATA PREPROCESSING

# we start by reading the hdf5 files to a x_train variable, and return the segments to a train_x variable
# this applies for the test and validate sets as well.

if len(sys.argv)<2:
    print("Correct use:python script.py <valid_dataset>")
    sys.exit()

dataset = sys.argv[1]
if dataset == "opp":
    path = os.path.join(os.path.expanduser('~'), 'Desktop', 'opportunity.h5')
else:
    print("Dataset not supported yet")
    sys.exit()

f = h5py.File(path, 'r')

x_train = f.get('train').get('inputs')[()]
y_train = f.get('train').get('targets')[()]

x_test = f.get('test').get('inputs')[()]
y_test = f.get('test').get('targets')[()]

print("x_train shape = ", x_train.shape)
print("y_train shape =",y_train.shape)
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("features shape, labels shape, each features mean, each features standard deviation")
print(x_test.shape, y_test.shape,
      np.mean(x_test), np.std(x_test))
print("x_test shape =" ,x_test.shape)
print("y_test shape =",y_test.shape)
print("the dataset is therefore properly normalised, as expected.")


print(np.unique(y_train))
print(np.unique(y_test))
unq = np.unique(y_test)

input_width = 23
if dataset == "opp":
    input_width = 23
    print("segmenting signal...")
    train_x, train_y = segment_opp(x_train,y_train,input_width)
    test_x, test_y = segment_opp(x_test,y_test,input_width)
    print("signal segmented.")
else:
    print("no correct dataset")
    exit(0)

print("train_x shape =",train_x.shape)
print("train_y shape =",train_y.shape)
print("test_x shape =",test_x.shape)
print("test_y shape =",test_y.shape)

# One-hot label conversion

train = pd.get_dummies(train_y)
test = pd.get_dummies(test_y)

train, test = train.align(test, join='inner', axis=1) # maybe 'outer' is better

train_y = np.asarray(train)
test_y = np.asarray(test)


print("unique test_y",np.unique(test_y))
print("unique train_y",np.unique(train_y))
print("test_y[1]=",test_y[1])
# test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
print("train_y shape(1-hot) =",train_y.shape)
print("test_y shape(1-hot) =",test_y.shape)


config = Config(train_x, test_x, dataset, input_width)

X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_classes])

pred_Y = LSTM_Network(X, config)

# Loss,optimizer,evaluation
l2 = config.lambda_loss_amount * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# Softmax loss and L2
with tf.name_scope('loss'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y))  # + l2
tf.summary.scalar('loss', cost)

optimizer = tf.train.AdamOptimizer(
    learning_rate=config.learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

training_epochs = 10
loss_over_time_train = np.zeros(training_epochs)
accuracy_over_time_train = np.zeros(training_epochs)
loss_over_time_test = np.zeros(training_epochs)
accuracy_over_time_test = np.zeros(training_epochs)
total_batches = train_x.shape[0] // config.batch_size
best_accuracy = 0.0
# Launch the graph
with tf.Session() as sess:
    # Init summary
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test')
    # sess.run(init)
    tf.global_variables_initializer().run()
    # Keep training until reach max iterations
    # cost_history = np.empty(shape=[0],dtype=float)
    i = 0
    for epoch in range(training_epochs):
        cost_history_train = np.empty(shape=[0],dtype=float)
        accuracy_history_train = np.empty(shape=[0],dtype=float)
        cost_history_test = np.empty(shape=[0],dtype=float)
        accuracy_history_test = np.empty(shape=[0],dtype=float)
        for step in range(total_batches):
            offset = (step * config.batch_size) % (train_y.shape[0] - config.batch_size)
            batch_x = train_x[offset:(offset + config.batch_size), :, :]
            batch_y = train_y[offset:(offset + config.batch_size), :]

            # print("batch_x shape =",batch_x.shape
            # print("batch_y shape =",batch_y.shape

            train_summary, _, c, acc = sess.run([merged, optimizer, cost, accuracy], feed_dict={X: batch_x, Y : batch_y})
            # Add into train_writer, view in tensorboard
            train_writer.add_summary(train_summary, i)

            cost_history_train = np.append(cost_history_train,c)
            accuracy_history_train = np.append(accuracy_history_train, acc)
            i += 1

        loss_over_time_train[epoch] = np.mean(cost_history_train)
        accuracy_over_time_train[epoch] = np.mean(accuracy_history_train)

        print("Epoch: {},".format(epoch) +
              "Train accuracy : {},".format(accuracy_over_time_train[epoch]) +
              "Train Loss : {}".format(loss_over_time_train[epoch]))
        # after every epoch, we test the model with the test data
        test_summary, pred_out, accuracy_out, loss_out = sess.run([merged, pred_Y, accuracy, cost], feed_dict={X: test_x, Y: test_y})
        loss_over_time_test[epoch] = loss_out
        accuracy_over_time_test[epoch] = accuracy_out
        best_accuracy = max(best_accuracy, accuracy_out)
        print("Epoch: {},".format(epoch) +
              "Test accuracy : {},".format(accuracy_out) +
              "Test Loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)
        # Add into test_writer, view in tensorboard
        test_writer.add_summary(test_summary, i)

        # Save the info into a file
        # merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/')

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
    print("Final test accuracy: {}".format(accuracy_out))

    # MORE METRICS
    print("Best epoch's test accuracy: {}".format(best_accuracy))
    # pred_Y is the result of the FF-RNN
    y_p = tf.argmax(pred_Y, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X:test_x, Y:test_y})
    print("Validation accuracy:", val_accuracy)
    y_true = np.argmax(test_y,1)

    if dataset=="opp":
        #print("f1_score_mean", metrics.f1_score(y_true, y_pred, average="micro")
        print("f1_score_weighted", metrics.f1_score(y_true, y_pred, average="weighted"))
        print("f1_score_macro", metrics.f1_score(y_true, y_pred, average="macro"))
        # print("f1_score_per_class", metrics.f1_score(y_true, y_pred, average=None)
    else:
        print("wrong dataset")

    plt.figure(1)
    #indep_train_axis = np.array(range(config.batch_size, (len(loss_over_time_train)+1)*config.batch_size, config.batch_size))
    plt.plot(loss_over_time_train,   "b--", label="Train losses")
    plt.plot(accuracy_over_time_train, "g--", label="Train accuracies")
    #indep_test_axis = np.array(range(config.batch_size, (len(loss_over_time_test)+1)*config.batch_size, config.batch_size))
    plt.plot(loss_over_time_test,     "b-", label="Test losses")
    plt.plot(accuracy_over_time_test, "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("confusion_matrix: /n", confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    with open('out_fflstm.txt', 'w') as f:
        #print >> f, 'Confusion Matrix: ', confusion_matrix
        #print >> f, 'Normalised Confusion Matrix: ', ormalised_confusion_matrix
        print('Confusion Matrix:', confusion_matrix, file=f)
        print('Normalised Confusion Matrix:', normalised_confusion_matrix, file=f)
    # Plot Results:
    # plt.figure(4)
    # plt.imshow(
    #     normalised_confusion_matrix,
    #     interpolation='nearest',
    #     cmap=plt.cm.rainbow
    # )
    # plt.title("Confusion matrix \n(normalised to % of total test data)")
    # plt.colorbar()
    # tick_marks = np.arange(18)
    # plt.xticks(tick_marks, label_map, rotation=90)
    # plt.yticks(tick_marks, label_map)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    #![png](LSTM_files/LSTM_16_0.png)
#######################################################################################
#### micro- macro- weighted explanation ###############################################
#                                                                                     #
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html      #
#                                                                                     #
# micro :Calculate metrics globally by counting the total true positives,             #
# false negatives and false positives.                                                #
#                                                                                     #
# macro :Calculate metrics for each label, and find their unweighted mean.            #
# This does not take label imbalance into account.                                    #
#                                                                                     #
# weighted :Calculate metrics for each label, and find their average, weighted        #
# by support (the number of true instances for each label). This alters macro         #
# to account for label imbalance; it can result in an F-score that is not between     #
# precision and recall.                                                               #
#                                                                                     #
#######################################################################################


print("--- %s seconds ---" % (time.time() - start_time))
print("Feed-forward LSTM Opportunity Done")
