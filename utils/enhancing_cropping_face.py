# Add GFPGAN folder to path
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)
gfpgan_dir = os.path.join(current_dir, 'GFPGAN')
sys.path.append(gfpgan_dir)

# Add neccessary library for the functions
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer

# ------------------------ Set up settings for GFPGAN ------------------------
if not torch.cuda.is_available():  # CPU
    import warnings
    warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                  'If you really want to use it, please modify the corresponding codes.')
    bg_upsampler = None
else:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=0, # Default 400
        tile_pad=10,
        pre_pad=0,
        half=True)  # need to set False in CPU mode

# ------------------------ Set up GFPGAN restorer ------------------------
arch = 'clean'
channel_multiplier = 2
model_path = os.path.join(parentdir, 'GFPGANv1.4.pth')
restorer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)

# ------------------------ Setup MTCNN and facenet model for face detection ------------------------
from facenet_pytorch import MTCNN, InceptionResnetV1
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# ------------------------ Create function to enhance face, then crop face further ------------------------
import math
def enhancing_cropping_face(img_path):
  img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
  # restore faces and background if necessary
  cropped_faces, restored_faces, restored_img = restorer.enhance(
    img_np_cvt,
    paste_back=True,
    weight=1)
  # Crop the faces further using MTCNN
  # By default, we expected the image only contain 1 face
  boxes, _ = mtcnn.detect(restored_faces[0]) # Gives the coordinates of the face in the given image
  cropped_enhanced_face = restored_faces[0][math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])] # Cropping the face
  return cropped_enhanced_face

# ------------------------ Create function to calculate distance of face vector embeddings ------------------------
from scipy.spatial.distance import cosine
import torchvision.transforms as transforms

def get_face_embedding(img_path):
    cropped_enhanced_face = enhancing_cropping_face(img_path)
    pil_to_tensor = transforms.ToTensor()(cropped_enhanced_face).unsqueeze_(0)
    image_embedding = resnet(pil_to_tensor)
    image_embedding = image_embedding.detach().numpy()
    return image_embedding

def get_similarity_score(img_path_1, img_path_2):
    face_embedding_1 = get_face_embedding(img_path_1)
    face_embedding_2 = get_face_embedding(img_path_2)
    score = cosine(face_embedding_1[0], face_embedding_2[0])
    return score