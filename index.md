---
layout: default
---

# LS-HDIB: A Large Scale Handwritten Document Image Binarization Dataset
<a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a>, <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a>, <a href="https://prajwalsingh.github.io/" target="_blank">Prajwal Singh</a>, <a href="https://people.iitgn.ac.in/~shanmuga/index.html" target="_blank">Shanmuganathan Raman</a>

**Accepted in ICPR 2022**.


<a href="https://arxiv.org/abs/2101.11674" target="_blank">Arxiv</a> link / <a href="https://github.com/kaustubh-sadekar/LS-HDIB" target="_blank">Code</a> link.

# Abstract

Handwritten document image binarization is challenging due to high variability in the written content and complex background attributes such as page style, paper quality, stains, shadow gradients, and non-uniform illumination. While the traditional thresholding methods do not effectively generalize on such challenging real-world scenarios, deep learning-based methods have performed relatively well when provided with sufficient training data. However, the existing datasets are limited in size and diversity. This work proposes LS-HDIB - a large-scale handwritten document image binarization dataset containing over a million document images that span numerous real-world scenarios. Additionally, we introduce a novel technique that uses a combination of adaptive thresholding and seamless cloning methods to create the dataset with accurate ground truths. Through an extensive quantitative and qualitative evaluation over eight different segmentation models, we demonstrate the enhancement in the performance of the deep networks when trained on the LS-HDIB dataset and tested on unseen images.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/teaser.png" />
</div>
> A few samples of handwritten document imagesobtained from the proposed LS-HDIB dataset with accurateground truths of the segmented foreground content obtainedusing our dataset generation method


# Dataset Details

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/bd-1.png" />
</div>
>  Block schematic of the proposed method for generating LS-HDIB dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/page_styles.png" />
</div>
>  A few sample images depicting different page styles available in LS-HDIB dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/bgd.png" />
</div>
>  A few sample images depicting different degradation effects available in LS-HDIB dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/content.png" />
</div>
>  A few sample images depicting the variation in the foreground content available in LS-HDIB dataset.


# Results 

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/hist-1.png" />
</div>
>  Quantitative comparison of performance of all the eight segmentation models over the three different test datasets when trained under Regime 1 (Blue), Regime 2 (Orange) and Regime 3 (Green).

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/qual_lshdib.png" />
</div>
>  Qualitative result on the LS-HDIB test dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/qual_bickley.png" />
</div>
>  Qualitative result on the Bickley Diary dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/qual_palm_leaf.png" />
</div>
>  Qualitative result on the Palm Leaf Manuscript dataset.

<div style="text-align:center">
    <img src="{{ site.baseurl }}/media/test_trend-1.png" />
</div>
>  Effect of varying dataset size on the model performance evaluated over the three test datasets.


# Citation

If you would like to cite us, kindly use the following BibTeX entry.

```
@misc{lshdib,
  author = {Sadekar, Kaustubh and Tiwari, Ashish and Singh, Prajwal and Raman, Shanmuganathan},
  title = {LS-HDIB: A Large Scale Handwritten Document Image Binarization Dataset},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
  doi = {10.48550/ARXIV.2101.11674},
}
```

# Acknowledgments
This research was supported by Science and Engineering Research Board (SERB) IMPacting Research INnovation and Technology (IMPRINT)-2 grant.

# Contact

Feel free to contact <a href="https://kaustubh-sadekar.github.io/" target="_blank">Kaustubh Sadekar</a>, <a href="https://www.linkedin.com/in/ashish-tiwari-82a392135/" target="_blank">Ashish Tiwari</a> or  <a href="https://prajwalsingh.github.io/" target="_blank">Prajwal Singh</a> for any further discussion about our work.

*Project page template inspired from [GradSLAM](https://gradslam.github.io/).*
