# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, get_person_scores

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'input_dir', '', 'Input directory')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')
    
tf.app.flags.DEFINE_integer(
    'max_imgs', 100, 'Maximum images to look at if a directory is specified')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    if FLAGS.frozen_model:
        model = load_graph(FLAGS.frozen_model)
    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

    scores = list()
    scores_wout_mispredictions = list()
    inference_time = list()
    
    if FLAGS.input_dir != '':        
        for root, dirs, files in os.walk(FLAGS.input_dir, topdown=False):
           counter = 0
           for name in tqdm(files):
              if counter >= FLAGS.max_imgs:
                break
           
              res, inf_time = get_score_from_image(os.path.join(root, name), gpu_options, config, model)
              
              if res > 0:
                scores_wout_mispredictions.append(res)
                
              scores.append(res)
              inference_time.append(inf_time)
                
              counter += 1
    else:
        res, inf_time  = get_score_from_image(FLAGS.input_img, gpu_options, config, model)
        
        if res > 0:
            scores_wout_mispredictions.append(res)
        
        scores.append(res)
        inference_time.append(inf_time)
        

    print("Average score across all images: " + str(np.mean(scores)))
    print("Average inference time: " + str(np.mean(inference_time)) + "ms")
    print("Average score disregarding mis-predictions: " + str(np.mean(scores_wout_mispredictions)))

              
def get_score_from_image(img_fp, gpu_options, config, model):
    img = Image.open(img_fp)
    img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(FLAGS.class_names)
    
    inference_start_time = time.time()
    if FLAGS.frozen_model:
        boxes, inputs = get_boxes_and_inputs_pb(model)

        with tf.Session(graph=model, config=config) as sess:
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        elif FLAGS.spp:
            model = yolo_v3.yolo_v3_spp
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})
                
    total_inference_time = time.time() - inference_start_time

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    return get_person_scores(filtered_boxes, classes), round(total_inference_time * 1000, 3)


if __name__ == '__main__':
    tf.app.run()
