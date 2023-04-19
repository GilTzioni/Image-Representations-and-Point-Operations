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
from ex1_utils import LOAD_GRAY_SCALE
import cv2

def gamma_correction(img , gamma):
    if(gamma!=0):
        inv_gamma = 1.0 / gamma
        table = (255.0 * (img / 255.0) ** inv_gamma).astype('uint8')
        return table
    return 0

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # Load the image
    img = cv2.imread(img_path)

    # Define the trackbar callback function
    def trackbar_callback(x):
        # Convert the trackbar value to gamma
        gamma = x / 100.0
        # Apply gamma correction to the image
        corrected_img = gamma_correction(img, gamma)
        # Display the corrected image
        cv2.imshow('Gamma Correction', corrected_img)

    # Create a window to display the image
    cv2.namedWindow('Gamma Correction')

    # Create the trackbar
    max_value = 200
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, max_value, trackbar_callback)
    # Display the initial image
    cv2.imshow('Gamma Correction', img)

    # Wait for a key event
    cv2.waitKey(0)

    # Destroy all windows
    cv2.destroyAllWindows()

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
