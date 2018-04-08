import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import urllib
import cv2
import time
import urllib.request

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

url = 'http://192.168.1.4:8080/video'

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')


# This is needed since the notebook is stored in the object_detection folder.



# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


filename = "/home/guru/practice/output11.avi"
# fourcc stands for four character code
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
framerate = 30
resolution = (640, 480)

VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)


# if you are fetting CUDA MEMORY erro, add this as config in session.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

bytes = b''
with detection_graph.as_default():
    with tf.Session(config=config, graph=detection_graph) as sess:

        stream = urllib.request.urlopen(url)

        while (True):
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                image_np = cv2.imdecode(np.fromstring(
                    jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                VideoFileOutput.write(image_np)
                cv2.imshow('live_detection', image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                    cv2.destroyAllWindows()
                    cap.release()
