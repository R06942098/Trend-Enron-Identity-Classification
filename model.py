import pandas as pd
import numpy as np
import tensorflow as tf 
import data_prepro


from layers import bidirectional_rnn, attention
from utils import get_shape, batch_doc_normalize
import tensorflow.contrib.layers as ly 
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
#email_df = pd.read_csv("enron_all.csv", na_filter= False) # Some empty subject may be set to value Nan.

class HAN_Classifier:

    def __init__(self, args):

        self.arg = args
        self.index_path = self.arg.index_path
        self.vocab_path = self.arg.vocab_path
        self.data_path = self.arg.data_path
        self.emb_path = self.arg.emb_path
        self.csv_path = self.arg.csv_path
        self.batch_size = self.arg.batch_size
        self.epo = self.arg.epoch
        self.embedding_size = self.arg.embedding_size
        self.dropout_rate = self.arg.dropout_rate
        self.cell_dim = self.arg.cell_dim
        self.att_dim = self.arg.att_dim
        self.learning_rate = self.arg.learning_rate

        self.email_df = pd.read_csv(self.csv_path, na_filter= False)
        self.trend_index = data_prepro.load_index(self.index_path)
        self.safe_index = data_prepro.safe_index
        data, label = data_prepro.process_whole_data_without_padding_tfidf(self.data_path, self.email_df, data_prepro.author_dict, self.safe_index)
        self.train_data, self.val_data, self.test_data, self.train_label, self.val_label, self.test_label = data_prepro.splitting_dataset(data, label, self.trend_index, self.safe_index)
        


        print("Training data size: {}.".format(self.train_data.shape))
        print("Testing data size: {}.".format(self.val_data.shape))
        print("Validation data size: {}.".format(self.test_data.shape))

        self.num_classes = max(label)+1

        #print(self.num_classes)
        self.pretrained_emb, self.vocab_size = data_prepro.embedding_matrix(self.emb_path+"glove.6B."+str(self.embedding_size)+"d.txt", self.vocab_path, self.email_df, self.embedding_size, self.safe_index)
        self.build_graph()


    def build_graph(self):
        self.docs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='docs')
        self.sent_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_lengths')
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')
        self.max_word_length = tf.placeholder(dtype=tf.int32, name='max_word_length')
        self.max_sent_length = tf.placeholder(dtype=tf.int32, name='max_sent_length')
        self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
        #self.dropout_rate = tf.placeholder(tf.float32, name="Dropout_prob")
        self.onehot_labels = tf.one_hot(self.labels, depth=self.num_classes)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.embedding_layer()
        self.word_levle_attention()
        self.sent_level_attention()
        self.classifier()

        ## Loss and Optimizer are listed as follow: 
        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.onehot_labels, logits= self.logits))
        predictions = tf.argmax(self.logits, axis=-1)
        #ground_truth = tf.argmax(self.onehot_labels, axis=-1)
        correct_preds = tf.equal(predictions, tf.cast(self.labels, tf.int64))
        self.batch_acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


        #theta_emb = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding')
        #self.assign = tf.assign(theta_emb, self.pretrained_emb)
        #theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=["word-level", "sent-level", "classifier"])


        ### Mechanism: Clipping gradient (for the exploration problom from LSTM)
        trained_vars = tf.trainable_variables()[1:]
        #print(trained_vars[:10])
        gradients = tf.gradients(self.cls_loss, trained_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars), name='train_op')


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def embedding_layer(self):
        with tf.variable_scope("embedding") as scope:
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.constant_initializer(self.pretrained_emb),
                dtype=tf.float32)

            self.embedded_inputs = tf.nn.embedding_lookup(
                        self.embedding_matrix, self.docs)

    def word_levle_attention(self):
        with tf.variable_scope('word-level') as scope:
            word_inputs = tf.reshape(self.embedded_inputs, [-1, self.max_word_length, self.embedding_size])
            word_lengths = tf.reshape(self.word_lengths, [-1])

            # word encoder
            cell_fw = rnn.GRUCell(self.cell_dim, name='cell_fw')
            cell_bw = rnn.GRUCell(self.cell_dim, name='cell_bw')
     
            init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                                      shape=[1, self.cell_dim],
                                                      initializer=tf.constant_initializer(0)),
                                      multiples=[get_shape(word_inputs)[0], 1])
            init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                                      shape=[1, self.cell_dim],
                                                      initializer=tf.constant_initializer(0)),
                                      multiples=[get_shape(word_inputs)[0], 1])

            rnn_outputs, _ = bidirectional_rnn(cell_fw=cell_fw,
                                                 cell_bw=cell_bw,
                                                 inputs=word_inputs,
                                                 input_lengths=word_lengths,
                                                 initial_state_fw=init_state_fw,
                                                 initial_state_bw=init_state_bw,
                                                 scope=scope)

            word_outputs, word_att_weights = attention(inputs=rnn_outputs,
                                                         att_dim=self.att_dim,
                                                         sequence_lengths=word_lengths)
            self.word_outputs = tf.layers.dropout(word_outputs, self.dropout_rate, training=self.is_training)

    def sent_level_attention(self):
        with tf.variable_scope('sent-level') as scope:
            sent_inputs = tf.reshape(self.word_outputs, [-1, self.max_sent_length, 2 * self.cell_dim])

            # sentence encoder
            cell_fw = rnn.GRUCell(self.cell_dim, name='cell_fw')
            cell_bw = rnn.GRUCell(self.cell_dim, name='cell_bw')

            init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                                  shape=[1, self.cell_dim],
                                                  initializer=tf.constant_initializer(0)),
                                  multiples=[get_shape(sent_inputs)[0], 1])
            init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                                  shape=[1, self.cell_dim],
                                                  initializer=tf.constant_initializer(0)),
                                  multiples=[get_shape(sent_inputs)[0], 1])

            rnn_outputs, _ = bidirectional_rnn(cell_fw=cell_fw,
                                             cell_bw=cell_bw,
                                             inputs=sent_inputs,
                                             input_lengths=self.sent_lengths,
                                             initial_state_fw=init_state_fw,
                                             initial_state_bw=init_state_bw,
                                             scope=scope)

            sent_outputs, sent_att_weights = attention(inputs=rnn_outputs,
                                                     att_dim=self.att_dim,
                                                     sequence_lengths=self.sent_lengths)
            self.sent_outputs = tf.layers.dropout(sent_outputs, self.dropout_rate, training=self.is_training)

    def classifier(self):
        ## output_layers: 
        with tf.variable_scope('classifier'):
            #self.logits = tf.layers.dense(inputs=self.sent_outputs, units=self.num_classes, name='logits')
            self.sent_outputs = ly.fully_connected(self.sent_outputs, 128)
            self.logits = ly.fully_connected(self.sent_outputs, self.num_classes, activation_fn=None)
            self.prob = tf.nn.softmax(self.logits)

    def get_feed_dict(self, docs, labels, training=False):
        padded_docs, sent_lengths, max_sent_length, word_lengths, max_word_length = batch_doc_normalize(docs, 30, 100)
        fd = {
          self.docs: padded_docs,
          self.sent_lengths: sent_lengths,
          self.word_lengths: word_lengths,
          self.max_sent_length: max_sent_length,
          self.max_word_length: max_word_length,
          self.labels: labels,
          self.is_training: training
        }
        """
        print(padded_docs.shape)
        print(sent_lengths.shape)
        print(max_sent_length.shape)
        print(word_lengths.shape)
        print(max_word_length)
        """

        return fd


    def next_batch(self, data, label, shuffle = False, batch_size = 256):
        le = len(data)
        epo = le // batch_size
        for i in range(0, le, batch_size):
            if i ==  (epo *batch_size) : 
                yield data[i:] , np.array(label[i:])
            else : 
                yield data[i: i+batch_size] , np.array(label[i: i+batch_size])



    def train(self):

        val_count = 0
        for epo in range(self.epo):
            for batch_docs, batch_labels in self.next_batch(self.train_data, self.train_label, batch_size=self.batch_size):
                _, _loss, _acc = self.sess.run([self.train_op, self.cls_loss, self.batch_acc],
                                         feed_dict=self.get_feed_dict(batch_docs, batch_labels, training=True))
                val_count += 1 
                #print(_loss)
                if val_count % 100 == 0:

                    val_loss = []
                    val_acc = []

                    for batch_docs, batch_labels in self.next_batch(self.val_data, self.val_label, batch_size=self.batch_size):
                        _loss, _acc, _prob = self.sess.run([self.cls_loss, self.batch_acc, self.prob],
                                         feed_dict=self.get_feed_dict(batch_docs, batch_labels))
                        val_loss.append(_loss)
                        val_acc.append(_acc)

                    print("Loss Validation: {}.".format(np.mean(val_loss)))
                    print("Acc Validation: {}.".format(np.mean(val_acc)))
                    print(batch_labels[:10])
                    print(np.argmax(_prob, axis=-1)[:10])
                #if epo % 2 == 0 :
                ## Evaluate using validation set! 


                ##self.saver.save()
    def test(self): 

        pass



