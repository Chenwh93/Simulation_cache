import numpy as np
import tensorflow as tf

TAU = 0.01  # soft replacement
GAMMA = 0.9
single_state = 1
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
single_action = 1


class DDPG(object):
    def __init__(self, sess, n_state, n_action, id, a_learning_rate=0.001, c_learning_rate=0.002):
        self.memory = np.zeros((MEMORY_CAPACITY, n_state * 2 + single_action + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = sess
        self.id = id

        self.n_state, self.n_action = n_state, n_action
        self.s = tf.placeholder(tf.float32, [None, n_state], 's')
        self.s_ = tf.placeholder(tf.float32, [None, n_state], 's_')
        self.r = tf.placeholder(tf.float32, None, 'r')

        a_scope_name = 'Actor' + str(id)
        with tf.variable_scope(a_scope_name):
            self.a = self._build_a(self.s, scope='eval', trainable=True)
            a_ = self._build_a(self.s_, scope='target', trainable=False)
        c_scope_name = 'Critic' + str(id)
        with tf.variable_scope(c_scope_name):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.s, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.s_, a_, scope='target', trainable=False)

        # networks parameters
        ae_scope = a_scope_name + '/eval'
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ae_scope)
        at_scope = a_scope_name + '/target'
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=at_scope)
        ce_scope = c_scope_name + '/eval'
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ce_scope)
        ct_scope = c_scope_name + '/target'
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ct_scope)

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.r + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(c_learning_rate).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(a_learning_rate).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        #probs = self.sess.run(self.a, {self.s: s})
        probs = self.sess.run(self.a, {self.s: s[np.newaxis, :]})
        #return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return np.argmax(probs.ravel())

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.n_state]
        ba = bt[:, self.n_state: self.n_state + self.n_action]
        br = bt[:, -self.n_state - 1: -self.n_state]
        bs_ = bt[:, -self.n_state:]

        self.sess.run(self.atrain, {self.s: bs})
        self.sess.run(self.ctrain, {self.s: bs, self.a: ba, self.r: br, self.s_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.n_action, activation=tf.nn.softmax, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.n_state, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_action, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
