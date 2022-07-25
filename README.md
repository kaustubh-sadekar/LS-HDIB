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

The output file will be saved as `<INPUT_FILE_NAME>_output.jpg`. For this specific example it will be input_2_output.jpg

*For better understanding of the input arguments type `python run.py -h`*


## Coming Soon
Easy to use google colab notebook


NOTE:

We used [segmentation_models_pytorch library](https://github.com/qubvel/segmentation_models.pytorch) for all the segmentation models. It has implementations for several segmentation models.
