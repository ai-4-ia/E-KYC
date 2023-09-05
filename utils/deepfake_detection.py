from functools import partial
import os
import inspect
import numpy as np
import torch
from torchvision.transforms import Normalize
from facenet_pytorch import MTCNN
import re
import cv2
import math
from timm.models.efficientnet import tf_efficientnet_b7_ns
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)


class DeepFakeDetection:
    '''A class to handle deepfake detection, the code is entirely based on the solution of selimsef: https://github.com/selimsef/dfdc_deepfake_challenge'''
    def __init__(self):
        self.mtcnn = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8])
        self.deepfake_detection_model = self.DeepFakeClassifier(encoder='tf_efficientnet_b7_ns')
        self.deepfake_detection_model_path = os.path.join(parentdir, 'models', 'final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40.pt')
        self.checkpoint = torch.load(self.deepfake_detection_model_path, map_location="cpu")
        self.state_dict = self.checkpoint.get("state_dict", self.checkpoint)
        self.deepfake_detection_model.load_state_dict({re.sub("^module.", "", k): v for k, v in self.state_dict.items()}, strict=True)
        self.deepfake_detection_model.eval()
        del self.checkpoint
    
    def put_to_center(self, img, input_size=380):
        img = img[:input_size, :input_size]
        image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        start_w = (input_size - img.shape[1]) // 2
        start_h = (input_size - img.shape[0]) // 2
        image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
        return image


    def isotropically_resize_image(self, img, size=380, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
        h, w = img.shape[:2]
        if max(w, h) == size:
            return img
        if w > h:
            scale = size / w
            h = h * scale
            w = size
        else:
            scale = size / h
            w = w * scale
            h = size
        interpolation = interpolation_up if scale > 1 else interpolation_down
        resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
        return resized
    
    def predict_deepfake(self, img_np):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize_transform = Normalize(mean, std)
        img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
        boxes, _ = self.mtcnn.detect(img_np_cvt) # Gives the coordinates of the face in the given image
        cropped_face = img_np_cvt[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])] # Cropping the face
        resized_face = self.isotropically_resize_image(cropped_face)
        resized_face = self.put_to_center(resized_face)
        
        # Processing tensor img for prediction
        face_tensor = torch.tensor(resized_face).float()
        face_tensor=face_tensor / 255.
        face_tensor=face_tensor.unsqueeze_(0)
        face_tensor = face_tensor.permute((0, 3, 1, 2))
        face_tensor = normalize_transform(face_tensor)
        
        # Make prediction
        y_pred = self.deepfake_detection_model(face_tensor)
        y_pred = torch.sigmoid(y_pred.squeeze())
        return y_pred.item()
    
    class DeepFakeClassifier(nn.Module):
        def __init__(self, encoder, dropout_rate=0.0) -> None:
            super().__init__()
            self.encoder_params = {
                "tf_efficientnet_b7_ns": {
                    "features": 2560,
                    "init_op": partial(tf_efficientnet_b7_ns, pretrained=False, drop_path_rate=0.2)
                },
            }
            # self.encoder = 'tf_efficientnet_b7_ns'
            self.encoder = self.encoder_params[encoder]["init_op"]()
            self.avg_pool = AdaptiveAvgPool2d((1, 1))
            self.dropout = Dropout(dropout_rate)
            self.fc = Linear(self.encoder_params[encoder]["features"], 1)

        def forward(self, x):
            x = self.encoder.forward_features(x)
            x = self.avg_pool(x).flatten(1)
            x = self.dropout(x)
            x = self.fc(x)
            return x