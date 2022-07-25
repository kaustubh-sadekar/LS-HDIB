# [LS-HDIB Dataset](https://kaustubh-sadekar.github.io/LS-HDIB/)
LS-HDIB: A Large Scale Handwritten Document Image Binarization Dataset

*Link to download initial dataset is [here](https://drive.google.com/drive/folders/1HSZ5j6dcl5LHJzoqRhBz06ZzvmfHAI7W?usp=sharing).*

## Setup and Install

```script
git clone https://github.com/kaustubh-sadekar/LS-HDIB.git
cd LS-HDIB/
pip install -r requirements.txt
```
## Run segmentation for your own image

```script
python run.py cpu input_2.jpg unet_best_weights.pth
```

*For better understanding of the input arguments type `python run.py -h`*
