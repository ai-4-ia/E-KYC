# Import necessary libraries
import os
import sys
import inspect
import torch
import cv2
import math
from facenet_pytorch import MTCNN
import warnings
# Add GFPGAN path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)
gfpgan_dir = os.path.join(current_dir, 'GFPGAN')
sys.path.append(gfpgan_dir)
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class FaceEnhancer:
    '''A class for initialize GFPGAN and perform face enhanced with cropping using MTCNN'''
    def __init__(self):
        # Init mtcnn for face detector
        self.mtcnn = MTCNN()
        # Init model for enhance background
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        # Init settings for GFPGAN
        if not torch.cuda.is_available():  # CPU
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                  'If you really want to use it, please modify the corresponding codes.')
            self.half = False # need to set False in CPU mode
        
        else:
            self.half = True
            
        self.bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=self.model,
            tile=0, # Default 400
            tile_pad=10,
            pre_pad=0,
            half=self.half)
        
        self.arch = 'clean'
        self.channel_multiplier = 2
        self.upscale = 2
        self.model_path = os.path.join(parentdir, 'models', 'GFPGANv1.4.pth')
        
        self.restorer = GFPGANer(
            model_path=self.model_path,
            upscale=self.upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler)

        
    def enhancing_and_cropping_face(self, img_np):
        '''Simply enhance using GFPGAN and further cropping the face with MTCNN for robust output'''
        img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
        # restore faces and background if necessary
        _, restored_faces, _ = self.restorer.enhance(
            img_np_cvt,
            paste_back=True,
            weight=1)
        # Crop the faces further using MTCNN
        # By default, we expected the image only contain 1 face
        boxes, _ = self.mtcnn.detect(restored_faces[0]) # Gets the coordinates of the face in the given image
        cropped_enhanced_face = restored_faces[0][math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), 
                                                  math.ceil(boxes[0][0]):math.ceil(boxes[0][2])] # Cropping the face
        return cropped_enhanced_face