import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Layer,Input,Embedding,Permute,Reshape,Conv2D,GlobalMaxPool2D,Lambda
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import  train_test_split
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#The coding method of word2vec is adopted in the example program, and the corresponding word vector can be trained according to the data set used

# create dict
def create_treshold_dict():
    threshold_dict={}
    with open('threshold/threshold.txt','r') as fr:
        for line in fr.readlines():
            key=line.split('\t')[0]
            upper=line.split('\t')[1]
            threshold_dict[key]=upper
    return threshold_dict

# get datasets
def dataset_embeding(rbp):
    word2vec_model = word2vec.Word2Vec.load("data/wv/word_vec_{}".format(rbp,rbp))
    pos_path = "data/sequence/sequence_pos/{}_pos".format(rbp)
    pos_list = seq2ngram(pos_path, 3, 1, word2vec_model)
    neg_path = "data/sequence/sequence_neg/{}_neg".format(rbp)
    neg_list = seq2ngram(neg_path, 3, 1, word2vec_model)
    seq_list = pos_list + neg_list
    feature = pad_sequences(seq_list, maxlen=TIME_STEP, padding="post",value=0)
    label = [1] * len(pos_list) + [0] * len(neg_list)
    embedding_matrix = np.zeros((len(word2vec_model.wv.vocab), EMBEDDING_DIM))
    for i in range(len(word2vec_model.wv.vocab)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return feature,label,embedding_matrix


"""
The squash function uses 0.5 instead of 1 in hinton's paper. If it is 1, the norm of all vectors will be reduced.
If it is 0.5, norms less than 0.5 will be reduced and norms greater than 0.5 will be enlarged
"""


def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()  # ||x||^2
    scale = K.sqrt(s_quared_norm) / (1 + s_quared_norm)  # ||x||/(0.5+||x||^2)
    result = scale * x
    return result


# Define our own softmax function instead of k.oftmax. Because k.oftmax cannot specify an axis
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result

# Define the margin loss by typing y_true, p_pred, returning the score, and passing fit
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    result = K.sum(y_true * K.square(K.relu(1 - margin - y_pred))
                   + lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    return result

def ndim(x):
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def expand_dims(x, axis=-1):
    return tf.expand_dims(x, axis)

def batch_dot(x, y, axes=None):
    py_any = any
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out

def seq2ngram(seq_path,k,s,model):
    # 打开序列文件
    with open(seq_path, "r") as fr:
        lines = fr.readlines()
    fr.close()
    list_full_text=[]
    for line in lines:
        if line.startswith(">") or len(line) < 3:
            continue
        else:
            line = line[:-1].upper()
            seq_len = len(line)
            list_line = []
            for index in range(0, seq_len, s):
                if index + k >= seq_len + 1:
                    break
                list_line.append(line[index:index+k])
            word_index=[]
            for word in list_line:
              if word in model.wv:
                  word_index.append(model.wv.vocab[word].index)
            list_full_text.append(word_index)
    return list_full_text

class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=2,
                 share_weights=False,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)


    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(  # [row,col,channel]->[1,input_dim_capsule,num_capsule*dim_capsule]
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)

    def call(self, inputs):
        if self.share_weights:
            # inputs: [batch, input_num_capsule, input_dim_capsule]
            # kernel: [1, input_dim_capsule, num_capsule*dim_capsule]
            # hat_inputs: [batch, input_num_capsule, num_capsule*dim_capsule]
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        # hat_inputs: [batch, input_num_capsule, num_capsule, dim_capsule]
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        # hat_inputs: [batch, num_capsule, input_num_capsule, dim_capsule]
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        # b: [batch, num_capsule, input_num_capsule]
        # 动态路由部分
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            o = self.activation(batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b += batch_dot(o, hat_inputs, [2, 3])
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def CapsNet(rbp,feature,label,embedding_matrix):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
    # bulid the model
    cap1_dim=8
    cap2_n=2
    cap2_dim=16
    inputs = Input(shape=(TIME_STEP,))
    embeding = Embedding(input_dim=embedding_matrix.shape[0],
                         output_dim=EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         trainable=True)(inputs)
    permute = Permute((2, 1))(embeding)
    cnn_inputs = Reshape((EMBEDDING_DIM, TIME_STEP, 1))(permute)
    cnn = Conv2D(filters=128, kernel_size=[EMBEDDING_DIM, 7], strides=1, padding='valid', activation='relu')(cnn_inputs)
    gmp = GlobalMaxPool2D()(cnn)
    caps_inputs = Reshape((-1, cap1_dim))(gmp)
    caps_outputs = Capsule(
        num_capsule=cap2_n, dim_capsule=cap2_dim,
        routings=2, share_weights=False)(caps_inputs)
    outputs = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(caps_outputs)
    model = Model(inputs=inputs, output=outputs)
    print(model.summary())
    print('\n****** Fit the model ******\n')
    # The EarlyStopping callback monitors training accuracy:
    # if it fails to improve for two consecutive epochs,training stops early
    # callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
    #                                                   monitor='val_loss',
    #                                                   save_best_only=True),
    #                   keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])

    # Fit the model
    y_train = np_utils.to_categorical(y_train, NUMCLASSES)
    y_test = np_utils.to_categorical(y_test, NUMCLASSES)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list,
                        validation_split=0.1, verbose=1)
    model.save('model/{}/{}_Caps_based.h5'.format(rbp,rbp))


    print('\n****** Check on test set ******\n')
    # evaluate the model on the test set
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\nLoss on test set: %0.4f' % score[0])
    print('\nAccuracy on test set: %0.4f' % score[1])


    print('\n****** Prediction on test set ******\n')
    # prediction
    y_pred_prob = model.predict(x_test)
    y_pred_label = np.argmax(y_pred_prob, axis=1)
    # take the class with the highest probability on the test set prediction
    y_test_label = np.argmax(y_test, axis=1)

    np.savetxt("metrics/CapsNet_based/{}/labels.txt".format(rbp), y_test_label)
    np.savetxt("metrics/CapsNet_based/{}/pred_probas.txt".format(rbp), y_pred_prob)
    auc = roc_auc_score(y_test_label, y_pred_prob[:, 1])
    print('AUC on Test Set: %.4f' % auc)
    acc = accuracy_score(y_test_label, y_pred_label)
    print('Accuracy on Test Set: %.4f' % acc)
    return acc, auc


if __name__ == "__main__":
    # hyper-parameter
    np.random.seed(42)
    tf.set_random_seed(42)
    EMBEDDING_DIM=32
    NUMCLASSES = 2
    BATCH_SIZE = 64
    EPOCHS = 30
    threshold_dict = create_treshold_dict()
    with open('metrics/metrics_caps.txt', 'w') as fw:
        fw.write("RBP\tAccuracy\tAUC\n")
        for rbp in threshold_dict.keys():
            TIME_STEP = int(threshold_dict[rbp])
            feature, label, embedding_matrix = dataset_embeding(rbp)
            # hyper-parameter
            # run the model
            acc, auc = CapsNet(rbp, feature, label, embedding_matrix)
            fw.write(rbp + "\t" + str(acc) + "\t" + str(auc) + "\n")
            fw.flush()
            print(rbp + " is finished!\n")