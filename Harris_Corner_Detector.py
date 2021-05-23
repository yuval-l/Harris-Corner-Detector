import cv2
import numpy as np
# import matplotlib.pyplot as plt


"""
----------------------------------------------------------------------------------------------------------------------
Yuval Levi & Ortal Michael
@HIT | Algorithms in multimedia and machine learning in the Python environment - Yakir Menahem
Spring 2021
----------------------------------------------------------------------------------------------------------------------

--- Harris corner detection implementation ---

Algorithm steps:
1. Compute image gradients: Gx,Gy
2. Compute Compute 2nd order moments: Gx*Gx, Gx*Gy, Gy*Gy
3. Filter products with a Gaussian window
Theoretical step: [For each pixel (i,j) define the matrix M]
4. Compute the score R
5. Threshold R and perform NMS (non-maxima suppression)

----------------------------------------------------------------------------------------------------------------------
"""


def compute_img_gradients(image):
    """
    :param image: grayscale input image
    :return: image x and y gradients
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    
    Gx = cv2.filter2D(image, cv2.CV_32F, sobel_x)
    Gy = cv2.filter2D(image, cv2.CV_32F, sobel_y)
    
    return Gx, Gy


def compute_products(Gx, Gy, k):
    """
    1. Computes 2nd order moments: Gx*Gx, Gx*Gy, Gy*Gy
    2. Filter products with Gaussian window
    :param Gx: Gradient by x axis
    :param Gy: Gradient by y axis
    :param k: size of window (kxk) to use in the filter (neighborhood area to check the gradients)
    :return: the filtered 2nd order moments
    """

    m11 = cv2.GaussianBlur(Gx**2, (k, k), 0)
    m22 = cv2.GaussianBlur(Gy**2, (k, k), 0)
    m12 = cv2.GaussianBlur(Gx*Gy, (k, k), 0)
    return m11, m22, m12


def R_threshold(R, th_ratio):
    """
    Thresholds the R-score image. Scalable to each image by using a th_ratio * image max value
    :param R: the R-score image
    :param th_ratio: threshold the image according to img max value * th_ratio
    :return: the threshold image of R-score
    """

    R_th = (R > R.max() * th_ratio) + 0
    return R_th


def detect_corners(img, **params):
    """
    The main corner detection function
    :param img: input image (BGR - since read using cv2.imread)
    :param params: dictionary with 3 parameters of the chosen sensitivity
            -  params['filter_size']: the size of the Gaussian window (neighborhood area to check the gradients)
            -  params['r_threshold']: the ratio to R-score image maximum value to use in order to threshold the image
            -  params['k_nms']: the kernel size to use for the NMS
    :return: RGB image with the corners found marked on it
    """

    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Compute image gradients: Gx,Gy
        Gx, Gy = compute_img_gradients(img_gray)

        # Steps 2+3: Gaussian filter on 2nd order moments
        m11, m22, m12 = compute_products(Gx, Gy, params['filter_size'])

        # Step 4: Score R:   det(M) - a * trace(M)^2
        traceM = m11 + m22
        detM = m11*m22-m12**2
        R = detM - 0.06 * (traceM**2)

        # Step 5: Threshold R and NMS
        R_th = R_threshold(R, params['r_threshold'])  # threshold
        R_dilate = cv2.dilate(R, np.ones((params['k_nms'], params['k_nms']))) #NMS
        R_nms = R >= R_dilate  # NMS
        R_final = R_th * R_nms  # Result after threshold and NMS

        # Draw found corners on colored image
        [y, x] = np.nonzero(R_final)
        for corner_y, corner_x in zip(y, x):
            cv2.circle(img, (corner_x, corner_y), 3, (0, 0, 255), -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.imshow("image with corners", img_rgb)
        # plt.imshow(img, cmap='gray')
        # plt.plot(x,y,'or');
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_rgb

    except Exception:
        print("HARRIS_CORNER_DETECTOR::ERROR")
