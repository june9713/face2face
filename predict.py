from model import Model
import pickle
import tensorflow as tf
from utils import plot

def main(FLAG):
    with open("./testImage.data" , 'rb') as f:
        testImage = pickle.load(f)
    with open("./testLabel.data" , 'rb') as f:
        testLabel = pickle.load(f)

    with open("./testAnswer.data" , 'rb') as f:
        testAnswer = pickle.load(f)

    with tf.Session() as sess:
        model = Model(sess, FLAG.input_dim, FLAG.hidden_dim, FLAG.output_dim)

        model.compile("mse", "adam")

        pred_y = model.predict(testImage[0:4])

        fig = plot(testImage[0:4], pred_y, True, "./predict.png")

def FLAG_parser():
    FLAG = tf.app.flags

    FLAG.DEFINE_integer("input_dim", 64*64, "size of input dimension")
    FLAG.DEFINE_integer("hidden_dim", 64, "dimension of hidden layer")
    FLAG.DEFINE_integer("output_dim", 8, "output dimension")
    FLAG.DEFINE_integer("width", 64, "image Width")
    FLAG.DEFINE_integer("height", 64, "image height")

    return FLAG.FLAGS


if __name__ == "__main__":
    FLAG = FLAG_parser()
    main(FLAG=FLAG)
