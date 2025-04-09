# Data Preparation

- MIMIC-CXR-JPG: https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/
- CheXpert: https://www.kaggle.com/datasets/ashery/chexpert

- NIH: https://www.kaggle.com/organizations/nih-chest-xrays/datasets
- VinDr-CXR: https://physionet.org/content/vindr-cxr/
- MedFMC: https://github.com/openmedlab/MedFM
- RSNA: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- SIIM：https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088

In this study, we clean the pre-training dataset by filtering out corrupted images and removing large black borders from the images.

Once downloaded the dataset, please change the data roots in `data_utils/data_path.py`.

