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
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import torchvision.models as models
import torch.nn as nn
from PIL import Image

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
model_path = os.path.join(parentdir, 'models', 'GFPGANv1.4.pth')
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
def enhancing_cropping_face(img_np):
    # img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
    # restore faces and background if necessary
    _, restored_faces, _ = restorer.enhance(
        img_np_cvt,
        paste_back=True,
        weight=1)
    # Crop the faces further using MTCNN
    # # By default, we expected the image only contain 1 face
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
    return image_embedding, cropped_enhanced_face

def get_similarity_score(img_path_1, img_path_2):
    face_embedding_1, cropped_enhanced_face_1 = get_face_embedding(img_path_1)
    face_embedding_2, cropped_enhanced_face_2 = get_face_embedding(img_path_2)
    score = cosine(face_embedding_1[0], face_embedding_2[0])
    return score, cropped_enhanced_face_1, cropped_enhanced_face_2

# ------------------------ Set up Midas model for depth map estimation ------------------------
def depth_map_estimation(img_np):
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    transform = midas_transforms.small_transform
    
    # img = cv2.imread("C:\\Users\\ADMIN\\Downloads\\PDV_CCCD.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
    boxes, _ = mtcnn.detect(img_np_cvt) # Gives the coordinates of the face in the given image
    
    input_batch = transform(img_np_cvt).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np_cvt.shape[:2],
            mode="bicubic",
            align_corners=False,
            ).squeeze()
    
    output = prediction.cpu().numpy()
    output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Colored Depth map
    output_norm = (output_norm*255).astype(np.uint8)
    output_norm = cv2.applyColorMap(output_norm, cv2.COLORMAP_VIRIDIS)
    output_face_area = output_norm[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])]
    return output_face_area
    
# ------------------------ Set up model for classify liveness face ------------------------
class CustomVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG16 = models.vgg16(pretrained=False)
        vgg16_num_ftrs = self.VGG16.classifier[6].in_features
        self.VGG16.classifier[6] = nn.Linear(vgg16_num_ftrs, 2)
    
    def forward(self, x_inp):
        x = self.VGG16(x_inp)
        return x

# ------------------------ Set up transform input for liveness classification ------------------------
# function to load image into required format for predictions
def tranform_img_from_pil(img_path):
    img = Image.open(img_path).convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
                        transforms.Resize(size=(224, 224)),    #VGG16 is trained on (224,224) images
                        transforms.ToTensor(),
                        normalize])
    img = img_transform(img)[:3,:,:].unsqueeze(0)
    return img

def transform_img_from_cv2(img_np):
    face_area_depth_map = depth_map_estimation(img_np)
    # print(f'face_area_depth_map: ', face_area_depth_map.shape)
    # face_area_depth_map = cv2.resize(face_area_depth_map, (224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
                        transforms.Resize(size=(224, 224)),    #VGG16 is trained on (224,224) images
                        transforms.ToTensor(),
                        normalize])
    img = img_transform(Image.fromarray(face_area_depth_map, 'RGB'))[:3,:,:].unsqueeze(0)
    # img = img_transform(face_area_depth_map)[:3,:,:].unsqueeze(0)
    return face_area_depth_map, img

def predict_liveness(img_np):
    # Load model
    loaded_model = CustomVGG16()
    model_path = os.path.join(parentdir, 'models', 'liveness_face.pt')
    loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # loaded_model.eval()
    
    # Create label array
    lbl_arrs = ['Not lived', "Lived"]
    
    # predict
    # face_area_depth_map, input_preprocessed = load_transform_image(img_np)
    
    # Currently there is slight error in transform img data directly
    # Hacky solution: write img temporarily then read again from PIL
    # After finish processing, then remove
    face_area_depth_map = depth_map_estimation(img_np)
    cv2.imwrite('temporary_processed_img.jpg', face_area_depth_map)
    img_path = os.path.join(parentdir, 'apps', 'temporary_processed_img.jpg')
    input_preprocessed = tranform_img_from_pil(img_path=img_path)
    with torch.no_grad():
        outputs = loaded_model(input_preprocessed)
        _, idx = torch.max(outputs, 1)
    os.remove(img_path)
    return face_area_depth_map, lbl_arrs[idx]