import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import cv2
import os
from PIL import Image
from scipy.ndimage import label

def foreground_pmap(img, fg_histogram, bg_histogram):
    h, w, c = img.shape
    n_bins = len(fg_histogram)
    binned_im = (img.astype(np.float32)/256*n_bins).astype(int)
    # 해당 픽셀의 r/g/b 값을 통해 (given color), 이게 fore일지 back일지에 대한 확률이 posterior probability 이다.
    # prior probabilities
    p_fg = 0.5
    p_bg = 1 - p_fg
    
    # extract fg & bg prob from histograms 
    p_rgb_given_fg = fg_histogram[binned_im[:, :, 0],
                                binned_im[:, :, 1], 
                                binned_im[:, :, 2]]
    
    p_rgb_given_bg = bg_histogram[binned_im[:, :, 0],
                                binned_im[:, :, 1],
                                binned_im[:, :, 2]]

    p_fg_given_rgb = (p_fg * p_rgb_given_fg /
                    (p_fg * p_rgb_given_fg + p_bg * p_rgb_given_bg))
    return p_fg_given_rgb

def unary_potentials(probability_map, unary_weight):
    ### BEGIN SOLUTION
    return -unary_weight * np.log(probability_map)

def calculate_histogram(img, mask, n_bins):
    histogram = np.full((n_bins, n_bins, n_bins), fill_value=0.001)
    
    ### BEGIN SOLUTION
    # convert values to range of bins
    binned_im = (img.astype(np.float32)/256*n_bins).astype(int)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y, x] != 0: # initial이든 진행중이든 
                histogram[binned_im[y, x, 0],
                        binned_im[y, x, 1], 
                        binned_im[y, x, 2]] += 1
                
    # normalize
    histogram /= np.sum(histogram)
    return histogram
    
def tile_pad(a, dims):
    return np.pad(a, tuple((0, i) for i in (np.array(dims) - a.shape)),
                mode='wrap')
def tile_rav_mult_idx(a, dims):
    return a.flat[np.ravel_multi_index(np.indices(dims), a.shape, mode='wrap')]


def get_ref(datafolder_loc, if_save=False, IMGH=1024, IMGW=1536, chann=3, save_folder=None):
    num_img=6
    out_img = np.zeros((IMGH, IMGW, chann))#final_out_ref
    rtoext = [IMGH//2,IMGW//num_img, chann]#region to which it is extracted

    def addtoout(if_fg, ser_num, limits, img_path):
        im = imageio.imread(img_path)
        im=im[:,:,:3]   
        h,w = im.shape[:2]
        init_fg_mask = np.zeros([h, w])
        init_bg_mask = np.zeros([h, w])
        init_fg_mask[limits[0]:limits[1], limits[2]:limits[3]] = 1
        init_bg_mask[limits[4]:limits[5], limits[6]:limits[7]] = 1
        fgh, fgw = limits[1]-limits[0], limits[3]-limits[2]#fgwidth, fgheight
        bgh, bgw = limits[5]-limits[4], limits[7]-limits[6]#fg
        # only_fg = im[init_fg_mask!=0][:].reshape(fgh, fgw, 3)
        patch = im[init_fg_mask!=0][:].reshape(fgh, fgw, 3) if if_fg else im[init_bg_mask!=0][:].reshape(bgh, bgw, 3)


        #created one image: half- possible fg, other half - possible bg
        #total 8 parts: one octant for each series (fg/bg)
        startr = 0 if if_fg else IMGH//2
        stopr = startr + rtoext[0]
        startc = rtoext[1]*ser_num
        stopc = startc + rtoext[1]
        out_img[startr:stopr, startc:stopc, :] = np.array(Image.fromarray(patch).resize((rtoext[1], rtoext[0]))) if patch.shape[0]>(stopr-startr)  else tile_rav_mult_idx(patch, rtoext) 

    # addtoout(False, ser_num, limits, img_path)
    imgslist = ["A3_plant009_rgb.png","A1_plant063_rgb.png","A3_plant023_rgb.png","A4_plant0029_rgb.png","A3_plant013_rgb.png","A3_plant021_rgb.png"]
    imgplist = [os.path.join(datafolder_loc, pitem) for pitem in imgslist]
    limits_list = [[1125, 1300, 830, 950, 1000, 2048, 1250, 2448],
    [240, 260, 100, 170, 0, 300, 380, 500],
    [590, 730, 1600, 1700, 0, 1000, 0, 1300],
    [200, 220, 210, 230, 250, 441, 0, 200],
    [1000,1050,1020,1050,400,1130,1730,2090], # Added
[640,660,1100,1150,250,2000,0,600]]
    #all four series

    if_fg = True
    for ser in range(num_img):
        addtoout(if_fg, ser, limits_list[ser], imgplist[ser]) 

    if_fg = False
    for ser in range(num_img):
        addtoout(if_fg, ser, limits_list[ser], imgplist[ser]) 

    if if_save:
        plt.figure()
        plt.imshow(out_img.astype(np.uint8))
        plt.savefig(os.path.join(save_folder, "refimg.png"))
        plt.close("all")
    return out_img

def crop_leafy(img_path=None, img=None, datafolder_loc=None, margin = 512, RESIZE_TO = [512,512], save_crop_img=False, if_save_ref=False, save_folder=None, output_image_name = "croppedimage.png"):   
    im = get_ref(datafolder_loc, if_save=if_save_ref, save_folder=save_folder)
    im=im[:,:,:3]
    h,w = im.shape[:2]
    init_fg_mask = np.zeros([h, w])
    init_bg_mask = np.zeros([h, w])
    init_fg_mask[0:512, :] = 1
    init_bg_mask[512:, :] = 1
    n_bins = 20
    fg_histogram = calculate_histogram(im, init_fg_mask, n_bins)
    bg_histogram = calculate_histogram(im, init_bg_mask, n_bins)

    im = imageio.imread(img_path)
    im=im[:,:,:3]

    foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram)
    # foreground_map = (foreground_prob > 0.5)

    # unary_weight = 1
    # unary_fg = unary_potentials(foreground_prob, unary_weight)
    # unary_bg = unary_potentials(1 - foreground_prob, unary_weight)

    thre=(foreground_prob>0.8).astype(np.bool)
    labeledimg, num_feat = label(thre)
    lab, cnts = np.unique(labeledimg, return_counts=True)
    lab1i, lab2i = np.argsort(cnts)[-1], np.argsort(cnts)[-2] 
    premask = np.zeros_like(im[:,:,0])
    #TODO generalize below...right now, selected second largest
    premask[labeledimg==lab[lab2i]]=1

    #A3-512, A1,A2,A4-120
    series = img_path.split("train_images")[-1][1:3]
    if series == "A3":
        margin = 512
    else:
        margin = 120
    rmin, rmax = premask.nonzero()[0].min(), premask.nonzero()[0].max()
    cmin, cmax = premask.nonzero()[1].min(), premask.nonzero()[1].max()
    croprmin = max(0, rmin-margin)
    croprmax = min(im.shape[0], rmax+margin)
    cropcmin = max(0, cmin-margin)
    cropcmax = min(im.shape[1], cmax+margin)
    new_img = im[croprmin: croprmax, cropcmin: cropcmax, :]
    crop_img = np.array(Image.fromarray(new_img).resize(RESIZE_TO))
    #bounding boxes, cropped img
    if save_crop_img:
        fig, axes = plt.subplots(1, 3, figsize=(10,5), sharey=True)
        axes[0].imshow(im)
        axes[1].imshow(premask)
        axes[2].imshow(crop_img)
        plt.savefig(os.path.join(save_folder, output_image_name))
        plt.close('all')
    return crop_img, [rmin, rmax, cmin, cmax]#cropped img with margin, bounding box without including margin


if __name__=="__main__":
    img_path_exp = "/home/students/thampi/PycharmProjects/leafcountchallenge/leafcountdata/LCC2020/train_images/A3_plant050_rgb.png"#did not work for 17  A2 18
    #loc_data = "/home/students/thampi/PycharmProjects/leafcountchallenge/leafcountdata/LCC2020/train_images"
    loc_data = "/home/students/thampi/PycharmProjects/leafcountchallenge/leafcountdata/LCC2020/train_images"
    save_folder = r"/home/students/thampi/PycharmProjects/leafcountchallenge/julichleafcount"
    crop_img, bng_boxes = crop_leafy(img_path = img_path_exp, datafolder_loc=loc_data, save_folder=save_folder, save_crop_img=True, if_save_ref=True)

