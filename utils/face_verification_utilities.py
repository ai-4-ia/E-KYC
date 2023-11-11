# Add GFPGAN folder to path
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)

# Add neccessary library for the functions
from facenet_pytorch import InceptionResnetV1
from face_enhancer import FaceEnhancer
import torchvision.transforms as transforms
from face_liveness_detection import FaceLivenessDetection
from deepfake_detection import DeepFakeDetection
from scipy.spatial.distance import cosine

# Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

class FacialAttributeVerification:
    def __init__(self):
        self.face_model = InceptionResnetV1(pretrained='vggface2').eval()
    
    def get_face_embedding(self, img_np):
        face_enhancer = FaceEnhancer()
        cropped_enhanced_face = face_enhancer.enhancing_and_cropping_face(img_np)
        pil_to_tensor = transforms.ToTensor()(cropped_enhanced_face).unsqueeze_(0)
        image_embedding = self.face_model(pil_to_tensor)
        image_embedding = image_embedding.detach().numpy()
        return image_embedding, cropped_enhanced_face

    def get_similarity_score(self, img_np_1, img_np_2):
        face_embedding_1, cropped_enhanced_face_1 = self.get_face_embedding(img_np_1)
        face_embedding_2, cropped_enhanced_face_2 = self.get_face_embedding(img_np_2)
        score = cosine(face_embedding_1[0], face_embedding_2[0])
        return score, cropped_enhanced_face_1, cropped_enhanced_face_2

def predict_liveness(img_np):
    face_liveness_prediction = FaceLivenessDetection()
    face_area_depth_map, label = face_liveness_prediction.predict_face_liveness(img_np)
    return face_area_depth_map, label


def deepfake_prediction(img_np):
    deepfake_detection = DeepFakeDetection()
    prediction = deepfake_detection.predict_deepfake(img_np)
    return prediction
