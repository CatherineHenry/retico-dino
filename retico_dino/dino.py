"""
DINO Module
==================

This module provides extracts features from DetectedObjectsIU using DINO.
"""


from collections import deque
from datetime import datetime

import numpy as np
import threading
import time
import torch
# from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as T
from PIL import Image

import retico_core
# TODO make is so that you don't need these 3 lines below
# idealy retico-vision would be in the env so you could 
# import it by just using:
# from retico_vision.vision import ImageIU, ExtractedObjectsIU
import sys
# prefix = '../../'
# sys.path.append(prefix+'retico-vision')

from retico_vision.vision import ExtractedObjectsIU, ObjectFeaturesIU

class Dinov2ObjectFeatures(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "DINOv2 Object Features"

    @staticmethod
    def description():
        return "Module for extracting visual features from images."

    @staticmethod
    def input_ius():
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu():
        return ObjectFeaturesIU

    def __init__(self, model_name = "dinov2_vits14", top_objects=1, show=False, save=False, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.top_objects = top_objects
        self.model = None
        self.feature_extractor = None
        self.show = show
        self.save = save
        self.queue = deque(maxlen=1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    # def get_clip_subimage(self, I, img_box):
    #     # expected format:
    #     # Numpy array, length 4, [xmin, ymin, xmax, ymax]

    #     xmin = int(img_box[0])
    #     xmax = int(img_box[2])
    #     ymin = int(img_box[1])
    #     ymax = int(img_box[3])
    #     sub = I.crop([xmin,ymin,xmax,ymax])

    #     if self.show:
    #         import cv2
    #         img_to_show = np.asarray(sub)
    #         cv2.imshow('image',cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) 
    #         cv2.waitKey(1)
    #     # pim = PImage.fromarray(sub)
    #     sub.load()
    #     return sub
    
    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_features = {}

            print("start dino processing")
            for i, sub_img in enumerate(detected_objects):
                if i>=self.top_objects: break
                # print(sub_img)
                sub = detected_objects[sub_img]
                if self.show:
                    import cv2
                    # img_to_show = np.asarray(sub)
                    cv2.imshow('image',sub)
                    cv2.waitKey(1)
                if self.save:
                    import cv2
                    img_to_save = np.asarray(sub)
                    cv2.imwrite(f"./test_dino_sub_img/{datetime.now().strftime('%m-%d_%H-%M-%S')}.jpg", img_to_save)
                # print(type(sub_img), type(detected_objects[sub_img]))
                # sub_img = self.get_clip_subimage(image, obj)


                # img = self.preprocess(sub_img).unsqueeze(0).to(self.device)
                # yhat = self.model.encode_image(img).cpu().numpy()
                # object_features[i] = yhat.tolist()
                # inputs = self.feature_extractor(images=sub_img_list, return_tensors="pt")
                # outputs = self.model(**inputs)
                # last_hidden_states = outputs.last_hidden_state
                img_tensor = self.feature_extractor(Image.fromarray(sub)).unsqueeze(0)#.to(self.device)
                feat = self.model(img_tensor).squeeze(0).detach().numpy().tolist()

                print(len(feat))
                object_features[i] = feat

            output_iu = self.create_iu(input_iu)
            output_iu.set_object_features(image, object_features)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        # self.model = ViTModel.from_pretrained(self.model_name)
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.feature_extractor = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()
    
    def shutdown(self):
        self._extractor_thread_active = False
