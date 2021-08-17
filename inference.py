"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

from reader import Reader
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from model import CycleGAN
import logging
import utils


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', './pretrained/random-standard2reallowdose75epochs.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input', '', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', '', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '512', 'image size, default: 512')

def inference():
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
    for i in range(3550,3750):
        FLAGS.input="./data/Results/RIDER CycleGANTestdata/RIDER CycleGANTestdata/"+str(i).rjust(6,'0')+".jpg"
        
        FLAGS.output="./data/Results/RIDER result real data 75 epochs/"+str(i).rjust(6,'0')+".jpg"
#        secondinput="./data/Results/highnoiseimages/original high noisy data/"+str(i).rjust(6,'0')+".jpg"
#        secondoutput="./data/Results/highnoisy100epochs/"+str(i).rjust(6,'0')+".jpg"
        with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
          image_data = f.read()
          input_image = tf.image.decode_jpeg(image_data, channels=1)
          input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
          input_image = utils.convert2float(input_image)
          input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 1])

#        with tf.gfile.FastGFile(secondinput, 'rb') as f2:
#          image_data2 = f2.read()
#          input_image2 = tf.image.decode_jpeg(image_data, channels=1)
#          input_image2 = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
#          input_image2 = utils.convert2float(input_image)
#          input_image2.set_shape([FLAGS.image_size, FLAGS.image_size, 1])

        [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')
#        [output_image2] = tf.import_graph_def(graph_def,
#                          input_map={'input_image': input_image2},
#                          return_elements=['output_image:0'],
#                          name='output')
        with tf.Session(graph=graph) as sess:
            generated = output_image.eval()
            #generated2 = output_image2.eval()
            with open(FLAGS.output, 'wb') as f:
              f.write(generated)
#            with open(secondoutput, 'wb') as f2:
#              f2.write(generated2)
        image_data=[]
        input_image=[]
        output_image=[]
        logging.info("%s" % FLAGS.input)
#        image_data2=[]
#        input_image2=[]
#        output_image2=[]
        

def main(unused_argv):
  inference()


if __name__ == '__main__':
  tf.app.run()
