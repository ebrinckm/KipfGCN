from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Empty')

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Helper function
def rohe(y_val):
    "Reverse One Hot Encoding"
    if(y_val[0] == 1):
        return 0
    else:
        return 1

def iter(labels):
    holder = np.zeros((len(labels)))
    for i, l in enumerate(labels):
        holder[i] = rohe(l)
    return holder

names = ['y']
y = []
for i in range(len(names)):
    with open("C:/Users/yello/Documents/Fall2020Intership/Clean/gcn-master/gcn/data/ind.{}.{}".format('Empty', names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            y = pkl.load(f, encoding='latin1')
        else:
            y = pkl.load(f)

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
# Initialize session
sess = tf.Session()

sess.run(tf.global_variables_initializer())
model.load(sess=sess)
cost_val = []
print('Attempting prediction...')
pred = model.predict()
print('...Prediction complete')

feed_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
pred = sess.run(pred, feed_dict=feed_val)

pred = pred[:1169631]
# Got the below line of code to translate confidence values to binary from:
# https://stackoverflow.com/questions/38654543/softmax-matrix-to-0-1-onehot-encoded-matrix

pred = tf.one_hot(tf.arg_max(pred, dimension=1), depth=2)
pred = sess.run(pred, feed_dict=feed_val)

print('Types:', type(y), type(pred))
print('Lengths:', len(y), len(pred))
pred = iter(pred)
y = iter(y)
print('Lengths:', len(y), len(pred))
print(y[:10])
print(pred[:10])

sum0 = 0
sum1 = 0
for i,l in enumerate(y):
    if(l == 1):
        sum0 += 1
    if(pred[i] == 1):
        sum1 += 1
print(sum0, sum1)

#res = precision_recall_fscore_support(y, pred)
res = f1_score(y, pred)
print(res)


