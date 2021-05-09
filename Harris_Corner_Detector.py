import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import os.path


def compute_img_gradients(img):
    sobel_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    Sobel_y = sobel_X.T
    
    Gx = cv2.filter2D(img, cv2.CV_32F, sobel_X)  
    Gy = cv2.filter2D(img, cv2.CV_32F, Sobel_y)
    
    return Gx, Gy


def compute_products(Gx, Gy, k):
    m11 = cv2.GaussianBlur(Gx**2, (k,k), 0)
    m22 = cv2.GaussianBlur(Gy**2, (k,k), 0)
    m12 = cv2.GaussianBlur(Gx*Gy, (k,k), 0)
    return m11, m22, m12


def R_threshold(img, R, th_ratio):
    R_th = (R > R.max()*th_ratio) + 0;
    return R_th


# def nms(G, theta):
#     # empty zeros matrice to store NMS
#     G_nms = np.zeros_like(G)
#
#     rows, cols = G.shape
#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#
#             direction = theta[i, j]
#
#             if (-112.5 < direction < -67.5) or (67.5 < direction < 112.5):  # N-S
#                 if (G[i, j] > G[i - 1, j]) and (G[i, j] > G[i + 1, j]):  # check in col
#                     G_nms[i, j] = G[i, j]
#
#             elif (-180 < direction < -157.5) or (157.5 < direction < 180) or (-22.5 < direction < 22.5):  # W-E
#                 if (G[i, j] > G[i, j - 1]) and (G[i, j] > G[i, j + 1]):  # check in row
#                     G_nms[i, j] = G[i, j]
#
#             elif (-157.5 < direction < -112.5) or (22.5 < direction < 67.5):  # NW-SE
#                 if (G[i, j] > G[i - 1, j - 1]) and (G[i, j] > G[i + 1, j + 1]):  # check in backslash
#                     G_nms[i, j] = G[i, j]
#
#             elif (-67.5 < direction < -22.5) or (112.5 < direction < 157.5):  # NE-SW
#                 if (G[i, j] > G[i - 1, j + 1]) and (G[i, j] > G[i + 1, j - 1]):  # check in slash
#                     G_nms[i, j] = G[i, j]
#
#     return G_nms


def main(img, **params):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Gx, Gy = compute_img_gradients(img_gray)
        m11, m22, m12 = compute_products(Gx, Gy, params['k_nms'])
        traceM = m11 + m22
        detM = m11*m22-m12**2
        R = detM - 0.06 * (traceM**2)
        R_th = R_threshold(img_gray, R, params['r_threshold'])

        R_dilate = cv2.dilate(R, np.ones((params['k_nms'],params['k_nms'])))
        R_nms = R >= R_dilate; # NMS

        # G = np.sqrt(Gx**2 + Gy**2)
        # theta = np.arctan2(Gy,Gx) / np.pi * 180
        # R_nms = nms(G, theta)

        R_final = R_th * R_nms # threshold and NMS

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
        print("ERROR")


if __name__ == '__main__':
    sg.theme('DarkBlue3')
    sg.theme('Light Grey 1')
    chosen_th = None

    low_sen_params = {
        'r_threshold': 0.1,
        'k_nms': 3
    }
    medium_sen_params = {
        'r_threshold': 0.05,
        'k_nms': 9
    }
    high_sen_params = {
        'r_threshold': 0.001,
        'k_nms': 13
    }

    control_column = [
        [sg.Text("Harris Corner Detector Simulator", size=(30,2), font=("Open Sans Bold", 18), text_color='dark blue')],
        [
            sg.Text("Choose image:"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FileBrowse()
        ],
        [
            sg.Radio("Low sensitivity", "Radio", size=(20, 1), key="-LOW-"),
        ],
        [
            sg.Radio("Medium sensitivity", "Radio", True, size=(20, 1), key="-MEDIUM-"),
        ],
        [
            sg.Radio("High sensitivity", "Radio", size=(20, 1), key="-HIGH-"),
        ],
        [sg.Button("Show Corners", key="-SHOW-")],
    ]

    image_viewer_column = [
        [sg.Text("Image Viewer", size=(30, 2), font=("Open Sans Bold", 12), text_color='dark blue')],
        [sg.Text(size=(50, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    layout = [
        [
            sg.Column(control_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ],
    ]

    window = sg.Window("Harris Corner Detector", layout, location=(800, 400))

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        elif event == "-FOLDER-":  # A file was chosen
            try:
                filename = values["-FOLDER-"]
                img = cv2.imread(filename)
                imgbytes = cv2.imencode(".png", img)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
            except:
                pass
        elif event == "-SHOW-":  # Show corners
            try:
                if values["-LOW-"]:
                    th_params = low_sen_params
                elif values["-MEDIUM-"]:
                    th_params = medium_sen_params
                elif values["-HIGH-"]:
                    th_params = high_sen_params

                filename = values["-FOLDER-"]
                img = cv2.imread(filename)
                img_corners = main(img, **th_params)
                imgbytes = cv2.imencode(".png", img)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
            except Exception:
                sg.popup_error(f'AN EXCEPTION OCCURRED!\nPlease load a valid image!')

    window.close()




