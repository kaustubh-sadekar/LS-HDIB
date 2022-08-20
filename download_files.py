import os
import sys
import gdown
import zipfile

data_folder = "."
os.makedirs(data_folder, exist_ok=True)
url = "https://drive.google.com/drive/folders/1gDZqhXsnAy1EmB5J55QNI_opINPjsEMd"
zip_fpaths = gdown.download_folder(url=url, output=data_folder, quiet=False, remaining_ok=False)
