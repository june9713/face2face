from model import Model
import tensorflow as tf
import pickle

def main(FLAG):
    with open("./trainImage.data" , 'rb') as f:
        trainImage = pickle.load(f)
    with open("./trainLabel.data" , 'rb') as f:
        trainLabel = pickle.load(f)

    with open("./testAnswer.data" , 'rb') as f:
        testAnswer = pickle.load(f)


    with tf.Session() as sess:
        model = Model(sess, FLAG.input_dim, FLAG.hidden_dim, FLAG.output_dim)

        model.compile("mse", "adam")

        model.fit(trainImage, trainLabel, EPOCH=FLAG.EPOCH, batch_size=FLAG.batch_size, learning_rate=FLAG.learning_rate)

def FLAG_parser():
    FLAG = tf.app.flags

    FLAG.DEFINE_float("learning_rate", 0.001, "learing_rate for train")
    FLAG.DEFINE_integer("input_dim", 64*64, "size of input dimension")
    FLAG.DEFINE_integer("hidden_dim", 64, "dimension of hidden layer")
    FLAG.DEFINE_integer("output_dim", 8, "output dimension")
    FLAG.DEFINE_integer("width", 64, "image Width")
    FLAG.DEFINE_integer("height", 64, "image height")
    FLAG.DEFINE_integer("batch_size", 64, "batch_size")
    FLAG.DEFINE_integer("EPOCH", 300, "EPOCH")

    return FLAG.FLAGS


if __name__ == "__main__":
    FLAG = FLAG_parser()
    main(FLAG=FLAG)
