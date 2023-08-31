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


# ------------------------ Set up model architecture for deep fake detection ------------------------
from functools import partial

import numpy as np
import torch
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

encoder_params = {
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
}


def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ------------------------ Preprocessing image for prediction  ------------------------
def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
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

# ------------------------ Load model and make deepfake detection  ------------------------
from torchvision.transforms import Normalize
import re

def load_deepfake_model():
    deep_fake_model_path = os.path.join(parentdir, 'models', 'final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40.pt')
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
    print("loading state dict {}".format(deep_fake_model_path))
    checkpoint = torch.load(deep_fake_model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    return model
    del checkpoint

deepfake_detection_model = load_deepfake_model()

def deepfake_prediction(img_np):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean, std)
    
    input_size = 380
    # define MTCNN face detect
    face_detect = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8])
    img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
    boxes, _ = face_detect.detect(img_np_cvt) # Gives the coordinates of the face in the given image
    cropped_face = img_np_cvt[math.ceil(boxes[0][1]):math.ceil(boxes[0][3]), math.ceil(boxes[0][0]):math.ceil(boxes[0][2])] # Cropping the face
    resized_face = isotropically_resize_image(cropped_face, input_size)
    resized_face = put_to_center(resized_face, input_size)
    
    # Processing tensor img for prediction
    face_tensor = torch.tensor(resized_face).float()
    face_tensor=face_tensor / 255.
    face_tensor=face_tensor.unsqueeze_(0)
    face_tensor = face_tensor.permute((0, 3, 1, 2))
    face_tensor = normalize_transform(face_tensor)
    
    # Make prediction
    y_pred = deepfake_detection_model(face_tensor)
    y_pred = torch.sigmoid(y_pred.squeeze())
    return y_pred.item()