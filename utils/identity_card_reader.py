import os
import inspect
import cv2
# import Craft class
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    empty_cuda_cache
)

import easyocr
import string
import random
import shutil

class IdentityCardReader:
    '''A class for initialize CRAFT and EASYOCR for document text recognition'''
    def __init__(self):
        # Load models to CPU
        self.refine_net = load_refinenet_model(cuda=False)
        self.craft_net = load_craftnet_model(cuda=False)
        self.craft_text_threshold = 0.7
        self.craft_link_threshold = 0.4
        self.craft_low_text = 0.4
        self.craft_long_size = 1280
        self.document_reader = easyocr.Reader(['vi'])
    
    def create_temporary_output_dir(self, output_dir_path):
        os.makedirs(output_dir_path)
        return
        
    def predict_text_area_and_output_temp_dir(self, img_np):
        # Convert img np and read image for CRAFT input format
        img_np_cvt = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Default cv2 return BGR mode
        img = read_image(image=img_np_cvt)
        # Perform text area prediction
        predicted_text_area_result = get_prediction(
            image=img,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.craft_text_threshold,
            link_threshold=self.craft_link_threshold,
            low_text=self.craft_low_text,
            cuda=False,
            long_size=self.craft_long_size
        )
        
        # Export to temporary output then perform ocr
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        chars = string.ascii_uppercase + string.digits
        random_temporary_dir_name = ''.join(random.choice(chars) for _ in range(10))
    
        while os.path.exists(os.path.join(current_dir, random_temporary_dir_name)):
            random_temporary_dir_name = ''.join(random.choice(chars) for _ in range(10))
        # os.makedirs(os.path.join(os.path.join(current_dir, random_temporary_dir_name)))
        self.create_temporary_output_dir(os.path.join(current_dir, random_temporary_dir_name))
        
        export_detected_regions(
            image=img,
            regions=predicted_text_area_result["boxes"],
            output_dir=os.path.join(current_dir, random_temporary_dir_name),
            rectify=True
        )
        
        # If use GPU, then unload empty GPU
        empty_cuda_cache()
        return os.path.join(current_dir, random_temporary_dir_name, "image_crops")
    
    def read_text_area(self, img_np):
        txt_detection_dict = {}
        
        craft_area_folder_path = self.predict_text_area_and_output_temp_dir(img_np=img_np)
        txt_img_names = os.listdir(craft_area_folder_path)
        for idx, _ in enumerate(txt_img_names):
            # if img_name == f"crop_{idx}.png" or img_name == "crop_9.png":
            img_name = f"crop_{idx}.png"
            if idx == 6 or idx == 9 or idx == 14:
                img_path = os.path.join(craft_area_folder_path, img_name)
                number_identity_txt = self.document_reader.readtext(img_path)
                txt_detection_dict[number_identity_txt[0][1]] = number_identity_txt[-1][1]
            # elif img_name == "crop_7.png":
            elif idx == 7 or idx == 11:
                img_path = os.path.join(craft_area_folder_path, img_name)
                name_img_path = os.path.join(craft_area_folder_path, f"crop_{idx+1}.png")
                name_holder = self.document_reader.readtext(img_path)
                name_value = self.document_reader.readtext(name_img_path)
                txt_detection_dict[name_holder[0][1]] =  r' '.join(i[1] for i in name_value)
            elif idx == 10:
                img_path = os.path.join(craft_area_folder_path, img_name)
                gender_nationality_text = self.document_reader.readtext(img_path)
                nationality_holder_value = gender_nationality_text[4][1].split(':')
                txt_detection_dict[gender_nationality_text[1][1]] = gender_nationality_text[2][1]
                txt_detection_dict[nationality_holder_value[0]] = nationality_holder_value[1]
            elif idx == 13:
                img_path = os.path.join(craft_area_folder_path, img_name)
                residency_img_path = os.path.join(craft_area_folder_path, f"crop_{idx+2}.png")
                residency_text = self.document_reader.readtext(img_path)[1][1]
                residency_holder_value = residency_text.split(":")
                residency_remaining_value = self.document_reader.readtext(residency_img_path)
                txt_detection_dict[residency_holder_value[0]] = residency_holder_value[1] + ' ' + \
                                                                r' '.join(i[1] for i in residency_remaining_value)
            else:
                continue
        parent_temp_dir = os.path.dirname(craft_area_folder_path)
        shutil.rmtree(parent_temp_dir)
        return txt_detection_dict
