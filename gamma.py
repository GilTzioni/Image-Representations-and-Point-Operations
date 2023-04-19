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
from __future__ import print_function
from __future__ import division
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import cv2

alpha_slider_max = 255
title_window = 'Gamma Correction'

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    #  adjusted gamma values
    global img
    if rep == 1:  # rep = Gray_Scale representation
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:  # =2, LOAD_RGB
        img = cv2.imread(img_path)

    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma %d' % alpha_slider_max
    cv2.createTrackbar(trackbar_name, title_window, 100, alpha_slider_max, trackBar)
    trackBar(0)
    print("Click on the screen to close the 'Gamma correction' window")
    # Keeping the window open until pressing any key
    cv2.waitKey()


def trackBar(n: int):
    alpha= float(n) / 100
    invGamma = 1000 if alpha == 0 else 1.0 / alpha
    max_ = 255
    table = np.array([((i / float(max_)) ** invGamma) * max_
                           for i in np.arange(0, max_ + 1)]).astype("uint8")
    img_ = cv2.LUT(img, table)
    cv2.imshow(title_window, img_)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()