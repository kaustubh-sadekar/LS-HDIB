import numpy as np
import cv2

import os
import glob

from vcam import vcam,meshGen
from tqdm import tqdm
import random


def fg_processing(img):

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,20)

	return thresh

def apply_texture(page, texture):
    width, height, channels = texture.shape
    center = (int(height/2), int(width/2))
    mask = 255 * np.ones(page.shape, page.dtype)
    mixedClone = cv2.seamlessClone(page, texture, mask, center, cv2.MIXED_CLONE)

    return mixedClone

def rotation_augment(img):

    rot1 = cv2.warpAffine(img, cv2.getRotationMatrix2D((240,240),90,1), (480,480))
    rot2 = cv2.warpAffine(img, cv2.getRotationMatrix2D((240,240),180,1), (480,480))
    rot3 = cv2.warpAffine(img, cv2.getRotationMatrix2D((240,240),270,1), (480,480))

    return img, rot1, rot2, rot3

def y_strech(img):

    c1 = vcam(480, 480)
    plane = meshGen(480, 480)
    plane.X *= 2
    plane.Y *= 0.7
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x,map_y = c1.getMaps(pts2d)
    output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_CUBIC, borderMode = cv2.BORDER_WRAP)
    output = cv2.flip(output,1)

    return output

def x_strech(img):

    c1 = vcam(480, 480)
    plane = meshGen(480, 480)
    plane.Y *= 2
    plane.X *= 0.7
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x,map_y = c1.getMaps(pts2d)
    output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_CUBIC, borderMode = cv2.BORDER_WRAP)
    output = cv2.flip(output,1)

    return output

def chalk_effect(img):

    c1 = vcam(480, 480)
    plane = meshGen(480, 480)
    plane.Z += 2.5*np.random.rand(480*480,1)
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x,map_y = c1.getMaps(pts2d)
    output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_CUBIC, borderMode = cv2.BORDER_WRAP)
    output = cv2.flip(output,1)

    return output

# Do not change the random seed to maintain reproducibility of the experiments
random.seed(43)
# Change the value to scale up the dataset size
NK_Value = 1
output_folder = "DataSet%dKHDIB"%NK_Value


try:
    os.mkdir(output_folder)
except:
    print(output_folder, " folder already exists!")
    exit(-1)

divisions = ['train', 'test', 'val']

for div in divisions:
    print("###########   ", div, "  ############")

    content_folder = "Content/%s/"%div
    page_folder = "Pages/%s/"%div
    texture_folder = "page_texture/%s/"%div

    content_paths = glob.glob(content_folder+"*")
    page_paths = glob.glob(page_folder+"*")
    texture_paths = glob.glob(texture_folder+"*")

    ################################################################
    #                        Texture generation                    #
    ################################################################

    texture_count = 1

    new_pages_path  = output_folder+"/Pages_%s"%div
    os.mkdir(new_pages_path)

    print("Number of original page images: ",len(page_paths))
    print("Number of original page texture images: ",len(texture_paths))
    print("Combining pages and textures to generate degraded document background....")

    for page_pth in tqdm(page_paths):
        pg = cv2.imread(page_pth)[80:-80,:,:].copy()
        cv2.imwrite(new_pages_path+"/pages_%d.jpg"%texture_count, pg)
        texture_count+=1
        random.shuffle(texture_paths)
        # for texture_pth in texture_paths[:9]:
        for texture_pth in texture_paths[:2]:
            txt = cv2.imread(texture_pth)[80:-80,:,:].copy()
            cv2.imwrite(new_pages_path+"/pages_%d.jpg"%texture_count, apply_texture(pg, txt))
            texture_count+=1
            

    output_path = output_folder+"/Input_%s"%div
    GT_path = output_folder+"/GT_%s"%div
    os.mkdir(output_path)
    os.mkdir(GT_path)
    augmented_texture_paths = glob.glob(new_pages_path+"/*")

    count = 1

    print("Combining written content pages and degraded pages to generate input (degraded handwritten document) and segmentation Ground Truth")
    for contpth in content_paths:
        ori_content = cv2.imread(contpth)
        ori_gt = fg_processing(ori_content)

        # cropping a 640x480 image into two 480x480 images
        cont1 = cv2.resize(ori_content[:320, :,:],(480,480)).copy()
        gt1 = cv2.resize(ori_gt[:320, :],(480,480)).copy()
        cont2 = cv2.resize(ori_content[-320:,:,:], (480,480)).copy()
        gt2 = cv2.resize(ori_gt[-320:,:], (480,480)).copy()

        cont1_rotations = rotation_augment(cont1)
        gt1_rotations = rotation_augment(gt1)
        cont2_rotations = rotation_augment(cont2)
        gt2_rotations = rotation_augment(gt2)

        for i in range(4):

            img = cont1_rotations[i]
            gt_img = gt1_rotations[i]
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img)
                count+=1

            img_wrapped = x_strech(img)
            gt_img_wrapped = x_strech(gt_img)
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img_wrapped , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img_wrapped)
                count+=1

            img_wrapped = y_strech(img)
            gt_img_wrapped = y_strech(gt_img)
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img_wrapped , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img_wrapped)
                count+=1

        for i in range(4):

            img = cont2_rotations[i]
            gt_img = gt2_rotations[i]
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img)
                count+=1
            
            img_wrapped = x_strech(img)
            gt_img_wrapped = x_strech(gt_img)
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img_wrapped , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img_wrapped)
                count+=1

            img_wrapped = y_strech(img)
            gt_img_wrapped = y_strech(gt_img)
            random.shuffle(augmented_texture_paths)
            random.shuffle(augmented_texture_paths)
            for aug_txt_pth in augmented_texture_paths[:NK_Value]:
                input_img = apply_texture(img_wrapped , cv2.imread(aug_txt_pth))
                cv2.imwrite(output_path+"/input_%d.jpg"%count, input_img)
                cv2.imwrite(GT_path+"/mask_%d.jpg"%count, gt_img_wrapped)
                count+=1
        print("Images Generated : ", count)
        if count> 200:
            break