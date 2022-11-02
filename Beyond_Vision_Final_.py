#Importing all Imp libraries

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util


#Preparation for downloading

#Pre-trained COCO model
MODEL_NAME='ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE= MODEL_NAME+'.tar.gz'
DOWNLOAD_BASE= 'http://download.tensorflow.org/models/object_detection/'
#Where to store it
PATH_TO_CKPT=  MODEL_NAME+ './frozen_inference_graph.pb'
#Label the path
PATH_TO_LABELS=os.path.join('data','mscoco_label_map.txt')
#Definind the number of classes 
NUM_CLASSES=90

#Checking if its already downloaded and if not, then download it
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading COCO Model...')
    opener=urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file=tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name=os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file,os.getcwd())
    print('COCO Model downloaded succesfully')
    
    
#If COCO Model has already been downloaded
else:
    print('COCO Model Already Exists')
    

#Loading the variables of frozen_inference_graph to use
detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph= fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')

#Loading Label Map
label_map = label_map_util.load_labelmap(os.path.join('F:\Beyond Vision\object_recognition_detection\data', 'mscoco_label_map.pbtxt'))

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



import cv2
img=cv2.VideoCapture(0)
#0 to capture only one frame, other values to capture other
state,pix=img.read()
loc='F:/Beyond Vision/pic'
#cv2.imshow('My Image',pix)
cv2.imwrite(os.path.join(loc,'img.jpg'),pix)
cv2.waitKey(0)
img.release()
cv2.destroyAllWindows()

#Converting images to numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

#To load the images for object detection
PATH_TO_TEST_IMAGES_DIR='F:/Beyond Vision/pic/'
TEST_IMAGE_PATH=[os.path.join(PATH_TO_TEST_IMAGES_DIR ,'img.jpg')]
#Defining size of output image
IMAGE_SIZE= (12,8)

# All variables for object detection and object detection operation
mainstr=[]
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATH:
            image=Image.open(image_path)
            image_np= load_image_into_numpy_array(image)
            image_np_expanded=np.expand_dims(image_np, axis=0)
            image_tensor= detection_graph.get_tensor_by_name('image_tensor:0') # Image
            boxes= detection_graph.get_tensor_by_name('detection_boxes:0')
            scores= detection_graph.get_tensor_by_name('detection_scores:0') #Match %
            classes= detection_graph.get_tensor_by_name('detection_classes:0') #Name of object
            num_detections= detection_graph.get_tensor_by_name('num_detections:0') #number of objects detected
            
            (boxes,scores,classes,num_detections)= sess.run([boxes,scores,classes,num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})
            
            #Visualization
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)
            """plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()"""
            
            
            
            
            final_score = np.squeeze(scores)    
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                    count = count + 1
            #print('cpunt',count)
            printcount =0;
            for i in classes[0]:
              printcount = printcount +1
              print(category_index[i]['name'])
              mainstr.append(category_index[i]['name'])
              
              if(printcount == count):
                    break

import pyttsx3
engine=pyttsx3.init()

if len(mainstr)<1:
    engine.say("Sorry No Object Was Detected!")
    engine.runAndWait()
elif len(mainstr)==1:
    engine.say("In front of you is a {}".format(mainstr))
    engine.runAndWait()
else:
    from collections import Counter
    x=Counter(mainstr)
    engine.say("In front of you are")
    engine.runAndWait()
    for i in x:
        if x[i]>1:
            object=i+"s"
        else:
            object=i
        engine.say("{} {}".format(x[i],object))
        engine.runAndWait()