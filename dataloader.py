import json
import os
import glob

from typing import Dict
from dataclasses import dataclass
import utils

import numpy as np
import pandas as pd
import requests
from xml.dom import minidom
from tqdm import tqdm

##Klasse, die die Bilder repräsentiert
@dataclass
class Input(object):
    id: int
    width: int
    height:int 
    file_name: str
    flickr_url: str

##Klasse, die die Annotationen repräsentiert
@dataclass
class Annotations(object):
    id: int
    image_id: int
    category_id: int
    bbox: list[int]

##Lädt alle Daten aus einer Json-Datei, die an dem übergebenen Pfad liegt.
##Die Daten werden in einer Liste von ImageWithAnn-Objekten verwaltet
@dataclass
class Data_class(object):
    image: Input
    anns: list[Annotations]

#format x1, y1, w, h
class DataLoader(object):
    def __init__(self, anns_path, tvt_path):
        #Pfade definieren
        ROOT_PATH = os.path.abspath('./')
        self.data_path = os.path.join(ROOT_PATH, 'data')
        self.anns_path = anns_path
        self.tvt_path = tvt_path
        self.img_path = os.path.join(self.data_path, 'images')
        self.labels_path = os.path.join(self.data_path, 'labels')

        #Annotations einlesen und in eigene Klassen umwandeln
        json_dct = self.load_json(anns_path)
        self.train = []
        self.val = []
        self.test = []
        list_images = []
        for i in json_dct["images"]:
            file_name = i["file_name"]
            file_name = file_name.replace("JPG", "jpg")
            list_images.append(Input(i["id"], i["width"], i["height"], 
                file_name, i["flickr_url"]))
    
        list_annotations = []
        for i in json_dct["annotations"]:
            list_annotations.append(Annotations(i["id"], i["image_id"], i["category_id"], i["bbox"]))
    
        list_Datapoints = []
        for i in list_images:
            list_anns = []
            for j in list_annotations:
                if i.id == j.image_id:
                    list_anns.append(j)
            list_Datapoints.append(Data_class(i, list_anns))
        
        self.datapoints = list_Datapoints

        #Direktories generieren und Bilder downloaden
        self.train_img_path = os.path.join(self.img_path, 'train_original')
        self.val_img_path = os.path.join(self.img_path, 'val_original')
        self.test_img_path = os.path.join(self.img_path, 'test_original')
        utils.create_dirs([self.data_path])
        utils.create_dirs([self.val_img_path])
        utils.create_dirs([self.train_img_path])
        utils.create_dirs([self.test_img_path])
        self.split_data_from_json()
        self.download(self.train, self.train_img_path)
        self.download(self.test, self.test_img_path)
        self.download(self.val, self.val_img_path)
    
    def download(self, data, path):
        for i in tqdm(data):
            if i.image.flickr_url is not None and not os.path.isfile(os.path.join(path, i.image.file_name)):
                r = requests.get(i.image.flickr_url, allow_redirects=True)
                if r.ok:
                    try:
                        data_path = os.path.join(path, i.image.file_name)
                        with open(data_path, 'wb') as f:
                            f.write(r.content)
                    except:
                        print(f'Error saving {os.path.basename(data_path)} to {os.path.dirname(data_path)}')
    
    ##lädt Daten aus json_file
    def load_json(self, data_path) -> Dict:
        try:
            with open(data_path, 'r') as f:
                json_dct = json.load(f)
                print(f'Successfully loaded {os.path.basename(data_path)} from {os.path.dirname(data_path)}')
                return json_dct
        except:
            print(f'Error loading {os.path.basename(data_path)} from {os.path.dirname(data_path)}')
            return None

     ##kreiert ein Dataframe (Tabelle) mit Bildnamen (ohne Indexe bzw mit falschen, alphabetisch geordnet)
    def create_df(self):
        all_img_paths = glob.glob(os.path.join(self.img_path, '*.jpg'))
        self.df = pd.DataFrame([os.path.basename(i) for i in all_img_paths], columns=['image_name'])

    ##spaltet die Daten in train, test und val und fügt diese zu der Dataframe-Tabelle hinzu
    ##Tabelle besteht dann aus Bildnamen und train-test-val-Zugehörigkeit (image_name und Sample)
    def split_data(self, shuffle=True, ratios=[0.6, 0.2, 0.2]):
        self.create_df()
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        total_length = self.df.shape[0]
        train_length = int(total_length * ratios[0])
        val_length = int(total_length * ratios[1])

        self.df['Sample'] = 'train'
        self.df['Sample'][train_length:] = 'val'
        self.df['Sample'][train_length+val_length:] = 'test'

        self.train = []
        self.test = []
        self.val = []

        for i in range(0, len(self.train)):
            if i < train_length:
                self.train.append(self.train[i])
            elif i < (train_length+val_length):
                self.val.append(self.train[i])
            else:
                self.test.append(self.train[i])

        num_train = self.df.loc[self.df['Sample'] == 'train'].shape[0]
        num_val = self.df.loc[self.df['Sample'] == 'val'].shape[0]
        num_test = self.df.loc[self.df['Sample'] == 'test'].shape[0]

        print(f'Split data into a training set (samples: {num_train}, percent: {num_train/total_length:.2f})')
        print(f'Split data into a val set (samples: {num_val}, percent: {num_val/total_length:.2f})')
        print(f'Split data into a test set (samples: {num_test}, percent: {num_test/total_length:.2f})')

        df_train = self.df.loc[self.df['Sample'] == 'train']
        df_test = self.df.loc[self.df['Sample'] == 'test']
        df_val = self.df.loc[self.df['Sample'] == 'val']

    def split_data_from_json(self):
        self.train = []
        self.test = []
        self.val = []
        tvt = self.load_json(self.tvt_path)
        jtrain, jval, jtest = tvt["train"], tvt["val"], tvt["test"]

        #datapoints in train, val, test-listen aufspalten
        for dp in tqdm(self.datapoints):
            found = False
            for i in range(0, len(jval)):
                if dp.image.file_name.__eq__(jval[i].replace("JPG", "jpg")):
                    self.val.append(dp)
                    found = True

            if not found:
                for i in range(0, len(jtest)):
                    if dp.image.file_name.__eq__(jtest[i].replace("JPG", "jpg")):
                        self.test.append(dp)
                        found =True

            if not found:
                for i in range(0, len(jtrain)):
                    if dp.image.file_name.__eq__(jtrain[i].replace("JPG", "jpg")):
                        self.train.append(dp)
                        found = True

def __main__():
    return
                    
if __name__ == '__main__':
    __main__()

