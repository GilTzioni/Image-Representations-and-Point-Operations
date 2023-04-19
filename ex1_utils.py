"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
ERROR_MSG = "Error: the given image has wrong dimensions"


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315203067


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Loading an image
    img = cv2.imread(filename)
    if img is not None:
        if representation == LOAD_GRAY_SCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif representation == LOAD_RGB:
            # We weren't asked to convert a grayscale image to RGB so this will suffice
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # Any other value was entered as the second parameter
            raise ValueError(ERROR_MSG)
    else:
        raise Exception("Could not read the image! Please try again.")
    return img / 255.0

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)

    # Determine the colormap to use for grayscale images
    cmap = 'gray' if representation == LOAD_GRAY_SCALE else None

    # Display the image using matplotlib's imshow function
    plt.imshow(image, cmap=cmap)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Define the transformation matrix
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.3111]])
    OrigShape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # taking sizes of input to make a new image
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.3111]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # from range [0,1] to [0,255]
    img = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (np.around(img)).astype('uint8')  # make sure its integers
    # original histogram
    histOrg = calHist(img)
    # cumsum
    cum_sum = histOrg.cumsum()

    lut = np.zeros(256)
    norm_cumSum = cum_sum / cum_sum.max()  # normalize each value of cumsum
    # create look up table
    for i in range(len(norm_cumSum)):
        new_color = int(np.floor(norm_cumSum[i] * 255))
        lut[i] = new_color

    imgEq = np.zeros_like(imgOrig, dtype=float)
    # Replace each intesity i with lut[i]
    for old_color, new_color in enumerate(lut):
        imgEq[img == old_color] = new_color

    # histogramEqualize
    histEQ = calHist(imgEq)
    # norm from range [0, 255] back to [0, 1]
    imgEq = imgEq / 255.0
    return imgEq, histOrg, histEQ

def calHist(img: np.ndarray) -> np.ndarray:  # My function to crate a Histogram
    hist = np.zeros(256)
    for pix in range(256):
        hist[pix] = np.count_nonzero(img == pix)

    return hist


def calcBoundaries(k: int):
    z = np.zeros(k + 1, dtype=int)
    size = 256 / k
    for i in range(1, k):  # first boundary 0
        z[i] = z[i - 1] + size
    z[k] = 255  # last boundary 255
    return z

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if (np.ndim(imOrig) == 2):  # its Gray scale image
        return quantize_chanel(imOrig.copy(), nQuant, nIter)

        # If an RGB image is given - convert it to YIQ
    rgb_to_yiq = transformRGB2YIQ(imOrig)
    # z : the borders which divide the histograms into segments
    qImage_i, MSE_error_list = quantize_chanel(rgb_to_yiq[:, :, 0].copy(), nQuant, nIter)  # take only Y chanel
    q_img_lst = []
    for img in qImage_i:
        # convert the original img back from YIQ to RGB
        qImage_tmp = transformYIQ2RGB(np.dstack((img, rgb_to_yiq[:, :, 1], rgb_to_yiq[:, :, 2])))  # Rebuilds rgb arrays
        q_img_lst.append(qImage_tmp)

    return q_img_lst, MSE_error_list


def quantize_chanel(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # return list:
    q_img_lst = []
    MSE_error_list = []

    # from range [0,1] to [0,255]
    img = imOrig * 255
    # create Histogram
    img_hist, edges = np.histogram(img.flatten(), bins=256)
    cumsum = img_hist.cumsum()
    z = calcBoundaries(nQuant)

    for iter in range(nIter):  # ask for make nIter times

        q = []  # contains the Weighted averages

        for i in range(nQuant):
            hist_range = img_hist[z[i]: z[i + 1]]
            i_range = range(len(hist_range))
            w_avg = np.average(i_range, weights=hist_range)
            q.append(w_avg + z[i])

        new_img = np.zeros_like(img)
        # Change all values in the range corresponding to the weighted average
        for border in range(len(q)):
            new_img[img > z[border]] = q[border]

        q_img_lst.append(new_img / 255.0)  # back to range [0,1]

        # Mean Squared Error
        MSE = np.sqrt((img - new_img) ** 2).mean()
        MSE_error_list.append(MSE)

        for i in range(1, len(q)):
            z[i] = (q[i - 1] + q[i]) / 2  # Change the boundaries according to q

    return q_img_lst, MSE_error_list