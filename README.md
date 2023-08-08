# E-KYC
# How to run
  1. pip install -r requirements.txt
  2. Move to utils/GFPGAN, execute: pip install -r requirements.text
  3. Run: python setup.py develop
  4. Download pretrained GFPGAN model using: wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
  5. Move pretrained model to models folder
  6. Move to apps folder
  7. Run: streamlit run main.py
