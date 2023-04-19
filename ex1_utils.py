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


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315203068


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    image_norm = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if representation == 1:
        ans_img = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)

    else:
        ans_img = cv2.cvtColor(image_norm, cv2.COLOR_BGR2RGB)
    return ans_img

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)

    # Determine the colormap to use for grayscale images
    cmap = 'gray' if representation == 1 else None

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
    A = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])

    # Flatten the image to a 2D array for matrix multiplication
    im_flat = imgRGB.reshape(-1, 3)

    # Convert the image to YIQ color space using matrix multiplication
    imYIQ_flat = np.dot(im_flat, A.T)

    # Reshape the image back to its original shape
    imYIQ = imYIQ_flat.reshape(imgRGB.shape)

    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Define the inverse transformation matrix
    A_inv = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.704]])

    # Flatten the image to a 2D array for matrix multiplication
    im_flat = imgYIQ.reshape(-1, 3)

    # Convert the image to RGB color space using matrix multiplication
    imRGB_flat = np.dot(im_flat, A_inv.T)

    # Reshape the image back to its original shape
    imRGB = imRGB_flat.reshape(imgYIQ.shape)

    return imRGB


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
    qImage_i, error = quantize_chanel(rgb_to_yiq[:, :, 0].copy(), nQuant, nIter)  # take only Y chanel
    qImage = []
    for img in qImage_i:
        # convert the original img back from YIQ to RGB
        qImage_tmp = transformYIQ2RGB(np.dstack((img, rgb_to_yiq[:, :, 1], rgb_to_yiq[:, :, 2])))  # Rebuilds rgb arrays
        qImage.append(qImage_tmp)

    return qImage, error


def quantize_chanel(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # return list:
    qImage = []
    Error = []

    # from range [0,1] to [0,255]
    img = imOrig * 255
    # create Histogram
    img_hist, edges = np.histogram(img.flatten(), bins=256)
    cumsum = img_hist.cumsum()
    # z = calcBoundaries(nQuant)
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

        qImage.append(new_img / 255.0)  # back to range [0,1]

        # Mean Squared Error
        MSE = np.sqrt((img - new_img) ** 2).mean()
        Error.append(MSE)

        for i in range(1, len(q)):
            z[i] = (q[i - 1] + q[i]) / 2  # Change the boundaries according to q

    return qImage, Error
    # # Check if the image is grayscale or RGB
    # is_gray = (np.ndim(imOrig) == 2)
    #
    # # Convert RGB image to YIQ
    # if not is_gray:
    #     imOrig = np.dot(imOrig, [0.299, 0.587, 0.114])
    #
    # # Initialize the segment division
    # hist, bins = np.histogram(imOrig.flatten(), 256, [0, 256])
    # seg_size = imOrig.size // nQuant
    # cumsum = np.cumsum(hist)
    # z = np.zeros(nQuant + 1)
    # for i in range(1, nQuant):
    #     z[i] = np.interp(i * seg_size, cumsum, bins[:-1])
    # z[-1] = 255
    #
    # # Perform nIter iterations
    # qImage_list = []
    # error_list = []
    # for i in range(nIter):
    #     # Find the values that each segment will map to
    #     q = np.zeros(nQuant)
    #     for j in range(nQuant):
    #         q[j] = np.average(imOrig[(z[j] <= imOrig) & (imOrig < z[j + 1])])
    #         # Quantize the image based on the current segment division and values
    #     qImage = np.zeros_like(imOrig)
    #     for j in range(nQuant):
    #        qImage[(z[j] <= imOrig) & (imOrig < z[j + 1])] = q[j]
    #
    #     # Calculate the total intensity error
    #     error = np.sum((imOrig - qImage) ** 2)
    #
    #     # Store the results
    #     qImage_list.append(qImage)
    #     error_list.append(error)
    #
    #     # Update the segment division for the next iteration
    #     hist, bins = np.histogram(qImage.flatten(), 256, [0, 256])
    #     cumsum = np.cumsum(hist)
    #     for j in range(1, nQuant):
    #          z[j] = np.interp(j * seg_size, cumsum, bins[:-1])
    #
    # # Convert quantized image back to RGB if necessary
    # if not is_gray:
    #     yiq = np.zeros_like(imOrig)
    #     yiq[:, :, 0] = qImage
    #     yiq[:, :, 1:] = imOrig[:, :, 1:]
    #     qImage_list = [np.dot(img, [[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]]) for img in qImage_list]
    #
    # return qImage_list, error_list

