import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)

from myTools import cnn_model as cnn

from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output/local_cnn/output_' + str(int(time.time())) + '.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    logging.info('load cat model :')
    model_cat = cnn.get_inception_v2_cat()
    logging.info('\tcat model loaded')

    logging.info('load dog model :')
    model_dog = cnn.get_inception_v2_dog()
    logging.info('\t dog model loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)

    t3 = time.time()
    cnn_output = cnn.get_more_data(img, model_cat, model_dog, (boxes, scores, classes, nums), class_names)
    t4 = time.time()

    logging.info('time: {}'.format(t2 - t1))

    logging.info('primary detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {:.2f}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)

    img = draw_outputs(img, (boxes, scores, classes, nums), class_names, cnn_output)

    cv2.imwrite(FLAGS.output, img)

    logging.info('secondary detections:')
    for dec2 in cnn_output:
        logging.info('\t{} : {}'.format(dec2[1], dec2[2]))

    logging.info('output saved to: {}'.format(FLAGS.output))
    cv2.imshow(FLAGS.output, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
