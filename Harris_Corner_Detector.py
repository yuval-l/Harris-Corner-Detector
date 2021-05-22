import cv2
import numpy as np
# import matplotlib.pyplot as plt


"""
Yuval Levi & Ortal Michael
@HIT | Algorithms in multimedia and machine learning in the Python environment - Yakir Menahem
Spring 2021


--- Harris corner detection implementation ---

Algorithm steps:
1. Compute image gradients: Gx,Gy
2. Compute Compute 2nd order moments: Gx*Gx, Gx*Gy, Gy*Gy
3. Filter products with a Gaussian window
Theoretical step: [For each pixel (i,j) define the matrix M]
4. Compute the score R
5. Threshold R and perform NMS (non-maxima suppression)

"""


def compute_img_gradients(image):
    """ Returns image x and y gradients """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    
    Gx = cv2.filter2D(image, cv2.CV_32F, sobel_x)
    Gy = cv2.filter2D(image, cv2.CV_32F, sobel_y)
    
    return Gx, Gy


def compute_products(Gx, Gy, k):
    """
    1. Computes 2nd order moments: Gx*Gx, Gx*Gy, Gy*Gy
    2. Filter products with Gaussian window
    Returns the filtered 2nd order moments
    """
    m11 = cv2.GaussianBlur(Gx**2, (k, k), 0)
    m22 = cv2.GaussianBlur(Gy**2, (k, k), 0)
    m12 = cv2.GaussianBlur(Gx*Gy, (k, k), 0)
    return m11, m22, m12


def R_threshold(img, R, th_ratio):
    """ Threshold image """
    R_th = (R > R.max() * th_ratio) + 0
    return R_th


def detect_corners(img, **params):
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
        R_th = R_threshold(img_gray, R, params['r_threshold'])  # threshold
        R_dilate = cv2.dilate(R, np.ones((params['k_nms'], params['k_nms']))) #NMS
        R_nms = R >= R_dilate  # NMS
        R_final = R_th * R_nms  # Result after threshold and NMS

        # Draw found corners on colored image
        [y,x] = np.nonzero(R_final)
        for corner_y, corner_x in zip(y, x):
            cv2.circle(img, (corner_x, corner_y), 3, (0, 0, 255), -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.imshow("image with corners", img_rgb)
        # plt.imshow(img, cmap='gray')
        # plt.plot(x,y,'or');
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_rgb

    except:
        print("HARRIS_CORNER_DETECTOR::ERROR")







