import tensorflow as tf
from tensorflow.contrib import layers
class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  #d_a=350
  #r=30
  def build_graph(self,usePreVector=False,vectors=None,n=20, d=50, u=32, d_a=32, r=1, reuse=False):
    with tf.variable_scope('SelfAttentive', reuse=reuse):
      # Hyperparmeters from paper
      self.n = n
      self.d = d
      self.d_a = d_a
      self.r = r
      regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
      initializer = tf.contrib.layers.xavier_initializer()
      # self.input_pl = tf.placeholder(tf.int32, shape=[None, self.n])
      self.input_pl = tf.placeholder(tf.int32, shape=[None, None])
      #zsy add 是否使用预训练好的词向量初始化
      if usePreVector==True:
        initial = tf.constant(vectors, dtype=tf.float32)
        embedding = tf.get_variable('word_vectors', initializer=initial)
      else:
        embedding = tf.get_variable('embedding', shape=[20000, self.d],
            initializer=initializer,regularizer=regularizer)


      input_embed = tf.nn.embedding_lookup(embedding, self.input_pl)
      # Declare trainable variables
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d, self.d],
          initializer=initializer,regularizer=regularizer)
      self.W_s2 = tf.get_variable('W_s2', shape=[self.d_a, self.d],
          initializer=initializer,regularizer=regularizer)

      self.W_s3 = tf.get_variable('W_s3', shape=[self.r, self.d_a],
          initializer=initializer,regularizer=regularizer)


      self.batch_size = batch_size = tf.shape(self.input_pl)[0]
      H = input_embed
      #类似双线性
      self.Q = Q = tf.nn.softmax(
              tf.map_fn(
                lambda x: tf.matmul(tf.matmul(x,self.W_s1),tf.transpose(x)),
                H))

      #点积
      # self.Q = Q = tf.nn.softmax(
      #         tf.map_fn(
      #           lambda x: tf.matmul(x,tf.transpose(x)),
      #           H))

      self.A = A = tf.matmul(Q,H)

      self.B = B = tf.nn.softmax(
        tf.map_fn(
          lambda x: tf.matmul(self.W_s3, x),
          tf.tanh(
            tf.map_fn(
              lambda x: tf.matmul(self.W_s2, tf.transpose(x)),
                  A))))


      self.M = tf.matmul(B, H)

      B_T = tf.transpose(B, perm=[0, 2, 1])
      tile_eye = tf.tile(tf.eye(r), [batch_size, 1])
      tile_eye = tf.reshape(tile_eye, [-1, r, r])
      BB_T = tf.matmul(B, B_T) - tile_eye
      self.P = tf.square(tf.norm(BB_T, axis=[-2, -1], ord='fro'))

  def trainable_vars(self):

    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
