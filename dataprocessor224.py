import os
import dataloader
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
from dataloader import *
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
from utils import area
from utils import showbbox

import cv2
import numpy as np
from numpy import asarray
from torchvision.io import read_image



anns_path = os.path.abspath(os.path.join('config', 'annotations', 'annotations.json'))
tvt_path = os.path.abspath(os.path.join( 'config', 'annotations', 'train_val_test_distribution_file.json'))
ROOT_PATH = os.path.abspath('./')
data_path = os.path.join(ROOT_PATH, 'data')
img_path = os.path.join(data_path, 'images')
train_img_path_original = os.path.join(img_path, 'train_original')
val_img_path_original = os.path.join(img_path, 'val_original')
test_img_path_original = os.path.join(img_path, 'test_original')
train_img_path_processed = os.path.join(img_path, 'train')
val_img_path_processed = os.path.join(img_path, 'val')
test_img_path_processed = os.path.join(img_path, 'test')
labels_path = os.path.join(data_path, 'labels')
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

#format x1, y1, x2, y2
class image_transformer(object):

    def __init__(self, data):
        self.data = data
        self.img_id = 772
        self.ann_id = 3718
        utils.create_dirs(os.path.join(data_path, '224', 'train', 'rubbish'))
        utils.create_dirs(os.path.join(data_path, '224', 'test', 'rubbish'))
        utils.create_dirs(os.path.join(data_path, '224', 'val', 'rubbish'))
        utils.create_dirs(os.path.join(data_path, '224', 'train', 'background'))
        utils.create_dirs(os.path.join(data_path, '224', 'test', 'background'))
        utils.create_dirs(os.path.join(data_path, '224', 'val', 'background'))
        self.img_id, self.ann_id, self.list_dataclass_train = self.process_image_mode(self.data.train, self.img_id, self.ann_id, train_img_path_original, train_img_path_processed, 15, 'train')
        self.img_id, self.ann_id, self.list_dataclass_val = self.process_image_mode(self.data.val, self.img_id, self.ann_id, val_img_path_original, val_img_path_processed, 15, 'val')
        self.img_id, self.ann_id, self.list_dataclass_test = self.process_image_mode(self.data.test, self.img_id, self.ann_id, test_img_path_original, test_img_path_processed, 15, 'test')


    def reduce(self, z1, z2, z3, z4, x1, y1):
        return z1-x1, z2-y1, z3-x1, z4-y1
    

    def calculate_anns(self, datapoint, x1, y1, x2, y2, ann_id, img_id, s):
        list_anns = []
        for i in range(0, len(datapoint.anns)):
            bbox = datapoint.anns[i].bbox
            ra = Rectangle(x1, y1, x2, y2)
            rb = Rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
            if area(ra, rb) is not None:
                if x1<=bbox[0]<=x2:
                    z1 = bbox[0]
                else:
                    z1 = x1
                if y1 <= bbox[1] <= y2:
                    z2 = bbox[1]
                else: 
                    z2 = y1
                if x1<=(bbox[0]+bbox[2])<=x2:
                    z3 = bbox[0]+bbox[2]
                else: 
                    z3 = x2
                if y1<=(bbox[1]+bbox[3])<=y2:
                    z4 = bbox[1]+bbox[3]
                else: 
                    z4 = y2
            
                z1,z2,z3,z4 = self.reduce(z1,z2,z3,z4, x1, y1)

                if not (z1==0 and (z3-z1)<s or z2==0 and (z4-z2)<s or z3==224 and (z3-z1)<s or z4==224 and (z4-z2)<s):
                    new_bbox = [z1, z2, z3, z4]
                    ann = Annotations(ann_id, img_id, 0, new_bbox)
                    list_anns.append(ann)
                    ann_id +=1
        return list_anns, ann_id

    def process_image(self, datapoint, im, img_id, ann_id, data_path_processed, list_dataclass, x1, y1, x2, y2, s, mode):
        image = im.crop((x1, y1, x2, y2)) 
        file_name = "{}_{}.jpg".format(datapoint.image.file_name[:-4], img_id) 
        input = Input(img_id, 224, 224, file_name, None)
        list_anns, ann_id = self.calculate_anns(datapoint, x1, y1, x2, y2, ann_id, img_id, s)
        img_id +=1
        data = Data_class(input, list_anns)
        list_dataclass.append(data)
        if len(data.anns) == 0:
            image.save(os.path.join(data_path, '224', mode, 'background', file_name))
        else:
            image.save(os.path.join(data_path, '224', mode, 'rubbish', file_name))
        return img_id, ann_id, list_dataclass


    def step_x(self, x1, x2):
        return x1+224, x2+224

    def step_y(self, y1, y2):
        return y1+224, y2+224

    def process_image_mode(self, datapoints, img_id, ann_id, data_path_original, data_path_new, s, mode):
        list_dataclass = []
        for i in range(0, len(datapoints)):
            datapoint = datapoints[i]
            path = os.path.join(data_path_original, datapoint.image.file_name)
            im = Image.open(path)
            width, height = datapoint.image.width, datapoint.image.height
            x1 = 0
            y1 = 0
            y2 = 224
            x2 = 224
            while ((x1+224) <= width):
                if ((y1+224) <= height):
                    img_id, ann_id, list_dataclass = self.process_image(datapoint, im, img_id, ann_id, data_path_new, list_dataclass, x1, y1, x2, y2, s, mode)
                    y1, y2 = self.step_y(y1, y2)
                else:
                    y2 = height
                    y1 = height-224
                    img_id, ann_id, list_dataclass = self.process_image(datapoint, im, img_id, ann_id, data_path_new, list_dataclass, x1, y1, x2, y2, s, mode)
                    y1, y2 = 0, 224
                    x1, x2 = self.step_x(x1, x2)
            x2 = width
            x1 = width-224
            y1, y2 = 0, 224

            while((y1+224) <= height):
                img_id, ann_id, list_dataclass = self.process_image(datapoint, im, img_id, ann_id, data_path_new, list_dataclass, x1, y1, x2, y2, s, mode) 
                y1, y2 = self.step_y(y1, y2) 
    
            x1, x2, y1, y2 = width-224, width, height-224, height
            img_id, ann_id, list_dataclass = self.process_image(datapoint, im, img_id, ann_id, data_path_new, list_dataclass, x1, y1, x2, y2, s, mode)
    
        return img_id, ann_id, list_dataclass
    
    def create_tvt_text(self):
        string = ''
        for i in range(0, len(self.list_dataclass_train)):
            string += self.list_dataclass_train[i].image.file_name
            string += '\n'
        path = os.path.join(labels_path, 'train.txt')
        with open(path, "w+") as f:
                f.write(string)

        string = ''
        for i in range(0, len(self.list_dataclass_val)):
            string += self.list_dataclass_val[i].image.file_name
            string += '\n'
        path = os.path.join(labels_path, 'val.txt')
        with open(path, "w+") as f:
                f.write(string)
        
        string = ''
        for i in range(0, len(self.list_dataclass_test)):
            string += self.list_dataclass_test[i].image.file_name
            string += '\n'
        path = os.path.join(labels_path, 'test.txt')
        with open(path, "w+") as f:
                f.write(string)
    
    def visualize_image(self):
        for i in range(0, len(self.list_dataclass_train)):
            if len(self.list_dataclass_train[i].anns)>0:
                showbbox(self.train_img_path, self.list_dataclass_train[i])


def __main__():
    return

if __name__ == "__main__":
    __main__()