""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

from reader import Reader
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python import pywrap_tensorflow
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints/20210420-0712', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'part_A2part_B.pb', 'XtoY model name, default: random-noisy2standard.pb')
tf.flags.DEFINE_string('YtoX_model', 'part_B2part_A.pb', 'YtoX model name, default: random-standard2noisy.pb')
tf.flags.DEFINE_integer('image_size', '512', 'image size, default: 512')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

def export_graph(model_name, XtoY=True):
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 1], name='input_image')
    cycle_gan.model()
    if XtoY:
      output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)
"""    
    latest_meta_path=latest_ckpt+'.meta'
    latest_meta= tf.train.import_meta_graph(latest_meta_path)
    graph=tf.get_default_graph()
    f = open('./data/Results/params2.txt','w')
    #for n in graph.as_graph_def().node:
    G_loss=graph.get_tensor_by_name('loss/G:0')
    print("HelloWorld!")
    print((G_loss))
    f.close()
   
    var_reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt)
    var_to_shape_map = var_reader.get_variable_to_shape_map()
    key='Variable_2'
    # Print tensor name and value
    f = open('./data/Results/params2.txt','w')
    #for key in var_to_shape_map:  # write tensors' names and values in file
        #print(key,file=f)
    print(latest_ckpt,file=f)
    print(key,file=f)
    print(var_reader.get_tensor(key),file=f)
    f.close()
"""


def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.XtoY_model, XtoY=True)
  print('Export YtoX model...')
  export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()
