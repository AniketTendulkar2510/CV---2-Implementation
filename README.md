# CV2-Implementation

The implementation can be tested by running the AttGANImplementtaion.ipynb file itself. Open the file in Google Collab to work with it.
This conatins the necessary steps for running the codes which are given in the zip file above.


The environment used is:

    Environment

    Python 3.6

    TensorFlow 1.15

    OpenCV, scikit-image, tqdm, oyaml

We recommend Anaconda or Miniconda, then you can create the AttGAN environment.
Steps to install the package have been mentioned in the Google Colab Notebook itself.


The dataset used is the CelebAHQ dataset. It has already been uploaded to google drive. The link is https://drive.google.com/file/d/1bCXiaG1ph1XB9sUWHsoCCO_EYf_DlEza/view?usp=sharing

We have already included the steps to use the dataset in the Google Collab Notebook. There is no need to download it separately.

The pre-trained AttGAN weights have already been uploaded to Google Drive. The steps to use them also have been mentioned in the notebook.
The link is: https://drive.google.com/file/d/1c4IEzya_bzKCOCHYDuV0U2YurB00Uzem/view?usp=sharing


Alternately, to run the environment on your personal computer if we have the required environment, we can use:
          conda create -n AttGAN python=3.6

          source activate AttGAN

          conda install opencv scikit-image tqdm tensorflow-gpu=1.15

          conda install -c conda-forge oyaml
          
          
Data Preparation

          CelebA-HQ (we use the data from CelebAMask-HQ, 3.2GB)

          CelebAMask-HQ.zip (move to ./data/CelebAMask-HQ.zip): https://drive.google.com/file/d/1bCXiaG1ph1XB9sUWHsoCCO_EYf_DlEza/view?usp=sharing

          unzip and process the data

          unzip ./data/CelebAMask-HQ.zip -d ./data/

          python ./scripts/split_CelebA-HQ.py
          
Run AttGAN

Training

        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
        --train_label_path ./data/CelebAMask-HQ/train_label.txt \
        --val_label_path ./data/CelebAMask-HQ/val_label.txt \
        --load_size 128 \
        --crop_size 128 \
        --n_epochs 200 \
        --epoch_start_decay 100 \
        --model model_128 \
        --experiment_name AttGAN_128_CelebA-HQ
        
        
Testing
 
Single Attributes

        CUDA_VISIBLE_DEVICES=0 \
        python test.py \
        --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
        --test_label_path ./data/CelebAMask-HQ/test_label.txt \
        --experiment_name AttGAN_128_CelebA-HQ
        
Multiple Attributes
        
        CUDA_VISIBLE_DEVICES=0 \
        python test_multi.py \
        --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
        --test_label_path ./data/CelebAMask-HQ/test_label.txt \
        --test_att_names Bushy_Eyebrows Pale_Skin \
        --experiment_name AttGAN_128
        
Attribute Sliding

        CUDA_VISIBLE_DEVICES=0 \
        python test_slide.py \
        --img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
        --test_label_path ./data/CelebAMask-HQ/test_label.txt \
        --test_att_name Pale_Skin \
        --test_int_min -2 \
        --test_int_max 2 \
        --test_int_step 0.5 \
        --experiment_name AttGAN_128

Pre-Trained Data
        
         https://drive.google.com/file/d/1c4IEzya_bzKCOCHYDuV0U2YurB00Uzem/view?usp=sharing
         AttGAN 128.zip
         
         unzip the file (AttGAN_128.zip for example)

        unzip ./output/AttGAN_128.zip -d ./output/
        
        Testing will be same
        
