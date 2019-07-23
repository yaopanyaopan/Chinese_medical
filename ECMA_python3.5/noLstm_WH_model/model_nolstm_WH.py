import tensorflow as tf

class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  #d_a=350
  #r=30
  #--------------------------ECMA － Attention1---没有加药物之间的关联处理-------------------
  def build_graph(self, n=20, d=50, d_a=32, r=1, reuse=False):
    with tf.variable_scope('SelfAttentive', reuse=reuse):
      # Hyperparmeters from paper
      self.n = n
      self.d = d
      self.d_a = d_a
      self.r = r
      regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
      initializer = tf.contrib.layers.xavier_initializer()

      embedding = tf.get_variable('embedding', shape=[3000, self.d],
          initializer=initializer)
      self.input_pl = tf.placeholder(tf.int32, shape=[None, self.n])
      input_embed = tf.nn.embedding_lookup(embedding, self.input_pl)

      # Declare trainable variables
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, self.d],
          initializer=initializer,regularizer=regularizer)
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
          initializer=initializer,regularizer=regularizer)

      # BiRNN
      self.batch_size = batch_size = tf.shape(self.input_pl)[0]
      H = input_embed
      self.A = A = tf.nn.softmax(
          tf.map_fn(
            lambda x: tf.matmul(self.W_s2, x),
            tf.tanh(
              tf.map_fn(
                lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                  H))))

      self.M = tf.matmul(A, H)

      A_T = tf.transpose(A, perm=[0, 2, 1])
      tile_eye = tf.tile(tf.eye(r), [batch_size, 1])
      tile_eye = tf.reshape(tile_eye, [-1, r, r])
      AA_T = tf.matmul(A, A_T) - tile_eye
      self.P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
