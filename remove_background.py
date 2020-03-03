import cv2
import argparse
import sys
import numpy as np
from pathlib import Path

def get_largest_bbox(image):
    """
    returns cropped image that fits largest contour found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    img_mask = cv2.inRange(img_gray, 1, 255)
    contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    return image[y:y+h,x:x+w,:]

def extract_contour(image):
    """
    Mask image using a binary threshold to create a mask and then using a bitwise_not application of said mask onto the original
    Arguments:
        image (cv2 image): image that will be masked out
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image, image, mask = mask_inv)

def blackout_background(image,threshold=.05):
    """
    Gets sorted list of unique colors counts; if pixel count exceeds certain porportion
    change this color to black (we can assume it is the background in this case)
    Arguments:
        image (cv2 image): image that will be blacked out
        threshold (float): value that determines which pixels are considered background
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unique_elements, counts_elements = np.unique(img_gray, return_counts=True)
    counts_sorted, unique_sorted = (list(t) for t in zip(*sorted(zip(counts_elements,
                                                                    unique_elements),reverse=True)))
    for i in range(len(counts_sorted)):
        if counts_sorted[i] < img_gray.size*threshold:
            break
        loc = np.where(img_gray==unique_sorted[i])
        image[loc] = 0
    return image
