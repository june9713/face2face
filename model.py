import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# train image size=64*64, train label rescaled to 0~1. value.
class Model:
    def __init__(self, sess, input_dim, hidden_dim, output_dim, load_model=True, model_path="./model/simpleModel.ckpt", name="face2face"):
        self.sess = sess

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = self.network(hidden_dim, output_dim, name)

        self.compiled = False

        self.model_path = model_path
        self.saver = tf.train.Saver()
        if os.path.exists("./model/checkpoint") and load_model == True:
            self.saver.restore(self.sess, self.model_path)
            print("Load Model Complete")

    def compile(self, loss, optimizer):
        try:
            loss = loss.lower()
            optimizer = optimizer.lower()
        except:
            print("enter loss, optimizer with string please")
            raise Exception

        #loss function
        if loss == "mse" or loss == "mean_squared_error":
            self.loss = tf.reduce_mean(tf.square(self.logits - self.y))
        elif loss == "softmax_cross_entropy":
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        elif loss == "sigmoid_cross_entropy":
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        #optimizer
        if optimizer == "adam":
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.loss)
        elif optimizer == "sgd":
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss=self.loss)
        elif optimizer == "rms" or optimizer == "rms_optimizer":
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss=self.loss)

        self.compiled = True

    def network(self, hidden_dim, output_dim, name):
        with tf.variable_scope(name):
            #모든 layer의 weight는 xavier initializer로 초기화하구 bias는 0.으로 초기화했습니다.
            relu1 = tf.layers.dense(self.x, units=hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(),
                                    bias_initializer=tf.constant_initializer(0.), name="_relu1")
            dropout1 = tf.nn.dropout(relu1, self.keep_prob, name="_dropout1")

            relu2 = tf.layers.dense(dropout1, units=hidden_dim*2, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(),
                                    bias_initializer=tf.constant_initializer(0.), name="_relu2")
            dropout2 = tf.nn.dropout(relu2, self.keep_prob, name="_dropout2")

            relu3 = tf.layers.dense(dropout2, units=hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(),
                                    bias_initializer=tf.constant_initializer(0.), name="_relu3")
            dropout3 = tf.nn.dropout(relu3, self.keep_prob, name="_dropout3")

            #마지막 output layer는 activation이 없습니다.
            logits = tf.layers.dense(dropout3, units=output_dim, activation=None, kernel_initializer=tf.truncated_normal_initializer(),
                                    bias_initializer=tf.constant_initializer(0.), name="_logits")

            return logits

    def fit(self, x, y, EPOCH, batch_size, learning_rate=0.001, save_model=True):
        if self.compiled == False:
            print("before training the model, compile first")
            raise Exception
        self.sess.run(tf.global_variables_initializer())
        cost_list = []

        n_sample = x.shape[0]
        step_for_epoch = int(n_sample/batch_size)

        for epoch in range(EPOCH):
            batch_idx = 0

            for step in range(step_for_epoch):
                batch_xs, batch_ys = x[batch_idx:batch_idx+batch_size], y[batch_idx:batch_idx+batch_size]
                batch_idx += batch_size

                feed_dict = {self.x:batch_xs, self.y:batch_ys, self.lr:learning_rate, self.keep_prob:0.75}
                _, cost = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)

                if step % 100 == 0:
                    print("Epoch:{}, Step:{}, COST:{}".format(epoch, step, cost))
                    cost_list.append(cost)

            if save_model == True:
                    if(not os.path.exists("./model")):
                        os.makedirs("./model")

                    save_path = self.saver.save(self.sess, self.model_path)
                    print("Epochs:{:01d}, Model saved in file{}".format(epoch, save_path))

        x = np.arange(len(cost_list))

        #train loss graph
        fig = plt.figure()
        plt.plot(x, cost_list, label='train_loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.ylim((0,1))

        fig.savefig("./loss_graph.png")
        print("Save Loss Graph")

        print("Show Loss Graph")
        plt.show()

    def predict(self, x):
        if self.compiled == False:
            print("before predict, compile first")
            raise Exception
        return self.sess.run(self.logits, feed_dict={self.x:x, self.keep_prob:1.0})


