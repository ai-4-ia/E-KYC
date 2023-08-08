# E-KYC
# How to run
  1. pip install -r requirements.txt
  2. Move to utils/GFPGAN, execute: pip install -r requirements.text
  3. Run: python setup.py develop
  4. Download pretrained GFPGAN model using: wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
  5. Move pretrained model to models folder
  6. Move to apps folder
  7. Run: streamlit run main.py
# Roadmap for implementation:
  1. Implement liveness detection on face images: first implement passive aproach using MIDAS for calculating dept mask estimation
  2. Implement identity card detection
  3. Implement deepfake detection
# Roadmap for improvements
  1. The pretrained model for extracting face embedding is trained using cross-entropy. The feature logits are not discriminative and thus are hard to distinguish different class(person). Need to fintune with triplet loss
  2. The pretrained model for extracting face embedding is trained with Caucassian faces, thus may not work well on Asian faces. Need to finetune with Asian dataset
  3. The pretrained model for extracting face embedding need to be finetuned with dataset after enhanced by GFPGAN or other GAN
  4. The pretrained model for extracting face embedding need to be finetuned with dataset from frontal face on identity card
