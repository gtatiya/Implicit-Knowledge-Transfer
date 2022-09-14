# Author: Gyan Tatiya

import numpy as np

import tensorflow as tf


def classifier(my_classifier, x_train_temp, x_test_temp, y_train_temp, y_test_temp):
    """
    Train a classifier on test data and return accuracy and prediction on test data
    :param my_classifier:
    :param x_train_temp:
    :param x_test_temp:
    :param y_train_temp:
    :param y_test_temp:
    :return: accuracy, prediction
    """

    # Fit the model on the training data.
    my_classifier.fit(x_train_temp, y_train_temp)

    # See how the model performs on the test data.
    accuracy = my_classifier.score(x_test_temp, y_test_temp)
    prediction = my_classifier.predict(x_test_temp)
    probability = my_classifier.predict_proba(x_test_temp)

    return accuracy, prediction, probability


############################### TensorFlow ###############################


class EncoderDecoderNetworkTF:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
    ):
        """
        Implement an encoder decoder network and train it
        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')

        self.code_prediction = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes ) +1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
            net = tf.layers.dense(inputs=net, units=self.n_dims_code)
        return net

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="decoder_" +str(i))
            net = tf.layers.dense(inputs=net, units=self.output_channels, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, epochs, logs_path, shuffle=False):
        """
        Train using provided data
        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path
        :param shuffle:
        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(epochs):

            if shuffle:
                random_idx = np.random.permutation(x_data.shape[0])
                x_data = x_data[random_idx]
                y_data = y_data[random_idx]

            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data})

            cost_log.append(c)

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate(self, x_data):
        """
        Generate target robot data using source robot data
        :param x_data: source robot data
        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error
        :param x_data:
        :param y_data:
        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        return loss
