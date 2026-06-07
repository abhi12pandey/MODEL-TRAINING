

!pip install -q kaggle tensorflow



import os
import zipfile



os.environ['KAGGLE_USERNAME'] = 'suhaib'
os.environ['KAGGLE_KEY'] = 'enter kaggle key'


# Download Dataset


!kaggle datasets download -d vipoooool/new-plant-diseases-dataset


with zipfile.ZipFile(
    "new-plant-diseases-dataset.zip",
    "r"
) as zip_ref:
    zip_ref.extractall("dataset")

print("DATASET DOWNLOADED & EXTRACTED SUCCESSFULLY")