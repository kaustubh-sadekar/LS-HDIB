---
layout: default
---

# LS-HDIB: A Large Scale Handwritten Document Image Binarization Dataset
<a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a>, <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a>, <a href="https://prajwalsingh.github.io/" target="_blank">Prajwal Singh</a>, <a href="https://people.iitgn.ac.in/~shanmuga/index.html" target="_blank">Shanmuganathan Raman</a>

# Abstract

Handwritten document image binarization is challenging due to high variability in the written content and complex background attributes such as page style, paper quality, stains, shadow gradients, and non-uniform illumination. While the traditional thresholding methods do not effectively generalize on such challenging real-world scenarios, deep learning-based methods have performed relatively well when provided with sufficient training data. However, the existing datasets are limited in size and diversity. This work proposes LS-HDIB - a large-scale handwritten document image binarization dataset containing over a million document images that span numerous real-world scenarios. Additionally, we introduce a novel technique that uses a combination of adaptive thresholding and seamless cloning methods to create the dataset with accurate ground truths. Through an extensive quantitative and qualitative evaluation over eight different segmentation models, we demonstrate the enhancement in the performance of the deep networks when trained on the LS-HDIB dataset and tested on unseen images.

# Results

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/teaser.png" />
</div>
> A few samples of handwritten document imagesobtained from the proposed LS-HDIB dataset with accurateground truths of the segmented foreground content obtainedusing our dataset generation method

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/bd.pdf" />
</div>
>  Block schematic of the proposed method for generating LS-HDIB dataset.

# Acknowledgments
This research was supported by Science and Engineering Research Board (SERB) IMPacting Research INnovation and Technology (IMPRINT)-2 grant.

# Contact

Feel free to contact <a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a>, <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a> or  <a href="https://prajwalsingh.github.io/" target="_blank">Prajwal Singh</a> for any further discussion about our work.
