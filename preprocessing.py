# %% preprocessing.py
import cv2
import numpy as np

#%% Gaussian blur
def restore_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

#%% Gamma correction
def enhance_image(image, gamma=1.5):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)

#%% Function to apply CLAHE to the image
""" def apply_clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab_image = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR) """

#%% Function to apply histogram equalization
""" def apply_histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR) """
