# Import necessary libraries
import os
import sys
import inspect
import cv2
import math
from facenet_pytorch import MTCNN
# Add GFPGAN path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)
gfpgan_dir = os.path.join(current_dir, 'GFPGAN')
sys.path.append(gfpgan_dir)
from gfpgan import GFPGANer


class FaceEnhancer:
    '''A class for initialize GFPGAN and perform face enhanced with cropping using MTCNN'''
    def __init__(self):
        # Init mtcnn for face detector
        self.mtcnn = MTCNN()
        # Init settings for GFPGAN
        self.bg_upsampler = None
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