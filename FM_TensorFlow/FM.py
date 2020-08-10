import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class FM:

    def __init__(self, n_feature, lr, l2_weight, latent_dim):
        self._parse_args(n_feature, lr, l2_weight, latent_dim)
        self._build_inputs()
        self._build_parameters()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, n_feature, lr, l2_weight, latent_dim):
        self.n_feature = n_feature
        self.lr = lr
        self.l2_weight = l2_weight
        self.latent_dim = latent_dim

    def _build_inputs(self):
        with tf.name_scope('inputs'):
            # batch_size * feature
            self.feature_ids = tf.placeholder(tf.int32, shape=[None, None], name='feature_ids')
            # batch_size * 1
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

    def _build_parameters(self):
        with tf.name_scope('parameters'):
            self.feature_embeddings = tf.Variable(
                tf.random_normal([self.n_feature, self.latent_dim], 0.0, 0.01),
                name='feature_embeddings')
            self.feature_biases = tf.Variable(
                tf.random_uniform([self.n_feature, 1], 0.0, 0.0),
                name='feature_biases')
            self.global_bias = tf.Variable(
                tf.constant(0.0),
                name='global_bias')

    def _build_model(self):
        with tf.name_scope('model'):
            # batch_size * feature * latent_dim
            batch_feature_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.feature_ids)
            # batch_size * latent_dim
            summed_feature_embeddings = tf.reduce_sum(batch_feature_embeddings, 1)
            summed_feature_embeddings_square = tf.square(summed_feature_embeddings)
            # batch_size * feature * latent_dim
            squared_feature_embeddings = tf.square(batch_feature_embeddings)
            # batch_size * latent_dim
            squared_sum_feature_embeddings = tf.reduce_sum(squared_feature_embeddings, 1)

            feature_comb_out = 0.5 * tf.subtract(summed_feature_embeddings_square, squared_sum_feature_embeddings)
            # batch_size * 1
            feature_comb_out = tf.reduce_sum(feature_comb_out, 1, keepdims=True)
            feature_bias_out = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_biases, self.feature_ids), 1)
            global_bias_out = self.global_bias * tf.ones_like(self.label)
            self.pred = tf.add_n([feature_comb_out, feature_bias_out, global_bias_out])

    def _build_loss(self):
        with tf.name_scope('loss'):
            base_loss = tf.nn.l2_loss(tf.subtract(self.label, self.pred))
            l2_loss = self.l2_weight * tf.nn.l2_loss(self.feature_embeddings)
            self.loss = base_loss + l2_loss

    def _build_train(self):
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def predict(self, sess, feed_dict):
        return self.pred.eval(feed_dict=feed_dict, session=sess)
