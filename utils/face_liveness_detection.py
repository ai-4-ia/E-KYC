import os
import inspect
import cv2
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import string
import random
# Add GFPGAN path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)


class FaceLivenessDetection:
    '''A class for handling main functions of liveness detection, has 2 inner class as 2 sub-components '''
    def __init__(self):
        self.depth_estimation_model = self.MiDaSModel()
        self.face_liveness_classification_model = self.CustomVGG16()
        self.liveness_classification_model_path = os.path.join(parentdir, 'models', 'liveness_face.pt')
        self.face_liveness_classification_model.load_state_dict(torch.load(self.liveness_classification_model_path, map_location='cpu'))
        self.face_liveness_classification_model.eval()
    
    class MiDaSModel:
        '''A sub-component for handling create depth map value of face images'''
        def __init__(self):
            self.mtcnn = MTCNN()   
            self.midas_model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            self.midas_model = torch.hub.load("intel-isl/MiDaS", self.midas_model_type)
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.midas_model.to(self.device)
            self.midas_model.eval()
            self.midas_model_transform = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_model_transform_type = self.midas_model_transform.small_transform
        
        def compute_depth_map_from_rgb_img(self, img_np):
            img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
            boxes, _ = self.mtcnn.detect(img_np_cvt) # Gives the coordinates of the face in the given image
            
            img_np_cvt_transformed = self.midas_model_transform_type(img_np_cvt).to(self.device)
            with torch.no_grad():
                prediction = self.midas_model(img_np_cvt_transformed)
                
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_np_cvt.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                    ).squeeze()
            output = prediction.cpu().numpy()
            output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Colored Depth map
            output_norm = (output_norm*255).astype(np.uint8)
            output_norm = cv2.applyColorMap(output_norm, cv2.COLORMAP_VIRIDIS)
            output_face_area = output_norm[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), 
                                           math.ceil(boxes[0][0]):math.ceil(boxes[0][2])]
            return output_face_area
    
    class CustomVGG16(nn.Module):
        '''A A sub-component train on depth map value for liveness prediction'''
        def __init__(self):
            super().__init__()
            self.VGG16 = models.vgg16(pretrained=False)
            vgg16_num_ftrs = self.VGG16.classifier[6].in_features
            self.VGG16.classifier[6] = nn.Linear(vgg16_num_ftrs, 2)
    
        def forward(self, x_inp):
            x = self.VGG16(x_inp)
            return x
    
    def tranform_img_from_pil(self, img_path):
        img = Image.open(img_path).convert('RGB')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_transform = transforms.Compose([
                            transforms.Resize(size=(224, 224)),    #VGG16 is trained on (224,224) images
                            transforms.ToTensor(),
                            normalize])
        img = img_transform(img)[:3,:,:].unsqueeze(0)
        return img

    def transform_img_from_cv2(self, img_np):
        face_area_depth_map = self.depth_estimation_model.compute_depth_map_from_rgb_img(img_np)
        # face_area_depth_map = cv2.resize(face_area_depth_map, (224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_transform = transforms.Compose([
                            transforms.Resize(size=(224, 224)),    #VGG16 is trained on (224,224) images
                            transforms.ToTensor(),
                            normalize])
        img = img_transform(Image.fromarray(face_area_depth_map, 'RGB'))[:3,:,:].unsqueeze(0)
        # img = img_transform(face_area_depth_map)[:3,:,:].unsqueeze(0)
        return img
    
    def predict_face_liveness(self, img_np):
        lbl_arrs = ['Not lived', "Lived"]
        face_area_depth_map = self.depth_estimation_model.compute_depth_map_from_rgb_img(img_np)
        
        chars = string.ascii_uppercase + string.digits
        random_temporary_img_name = ''.join(random.choice(chars) for _ in range(10)) + '.jpg'
        cv2.imwrite( random_temporary_img_name, face_area_depth_map)
        img_path = os.path.join(parentdir, 'apps', random_temporary_img_name)
        input_preprocessed = self.tranform_img_from_pil(img_path=img_path)
        with torch.no_grad():
            outputs = self.face_liveness_classification_model(input_preprocessed)
            _, idx = torch.max(outputs, 1)
        os.remove(img_path)
        return face_area_depth_map, lbl_arrs[idx]
