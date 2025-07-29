from skimage import io, filters, segmentation, morphology, measure, data, restoration, util
from skimage.color import rgb2gray
from skimage.util import crop
from skimage.filters import meijering, sato, frangi, hessian, apply_hysteresis_threshold
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
import numpy as np
import scipy.ndimage as ndi
import napari
import skimage as ski
import cv2
import sknw
import medpy
from medpy.filter.smoothing import anisotropic_diffusion
import networkx as nx

def wo_background(img) :
    background = restoration.rolling_ball(img, radius = 15)
    img = img - background
    return img

def contraste(img) :
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    cl = clahe.apply(l_channel)
    img = cv2.merge((cl,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def extract(photo, min_size = 5000, threshold_low = 0.05, threshold_high = 0.7, radius = 5) :
    # 1. Image
    img = io.imread(photo)
    img = img[:,:,1]

    # 2. Traitement 
    img = wo_background(img)  #retirer le fond
    img = contraste(img) #augmenter le contraste
    img = anisotropic_diffusion(img, niter = 5, kappa = 50, gamma = 0.1) #filtre anisotropique
    img = meijering(img, black_ridges = False, sigmas = range(1, 3)) #amélioration des edges
    img = apply_hysteresis_threshold(img, low = threshold_low, high = threshold_high) #image binaire
    img = morphology.remove_small_objects(img, min_size = min_size) #retirer les petits objets
    footprint = morphology.disk(radius = radius) #taille du cerle pour le closing
    img = morphology.binary_closing(img, footprint = footprint) #closing
    skeleton = skeletonize(img)

    # 3. Extraction du réseau
    G = sknw.build_sknw(skeleton)  

    return G







