from Harris_Corner_Detector import detect_corners
import cv2
import numpy as np
import PySimpleGUI as sg

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


Run the program:
- Load an image using "browse"
- Choose desired sensitivity level (low/medium/high) [default: medium]
- Click "show corners" to present results
- You can switch sensitivities at each point, the results will be updated only after re-clicking "show corners"
 
"""

if __name__ == '__main__':
    # GUI - set theme
    sg.theme('Light Grey 1')

    # Init threshold
    chosen_th = None

    # Dicts defining low/medium/high sensitivities thresholds
    low_sen_params = {
        'r_threshold': 0.1,
        'filter_size': 3,
        'k_nms': 13
    }
    medium_sen_params = {
        'r_threshold': 0.05,
        'filter_size': 7,
        'k_nms': 9
    }
    high_sen_params = {
        'r_threshold': 0.001,
        'filter_size': 11,
        'k_nms': 5
    }

    # GUI - define control panel - load image, set sensitivity, show corners
    control_column = [
        [sg.Text("Harris Corner Detector Simulator", size=(30,2), font=("Open Sans Bold", 18), text_color='dark blue')],
        [sg.Text("Choose image:"),
         sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
         sg.FileBrowse()],
        [sg.Radio("Low sensitivity", "Radio", size=(20, 1), key="-LOW-"),],
        [sg.Radio("Medium sensitivity", "Radio", True, size=(20, 1), key="-MEDIUM-"),],
        [sg.Radio("High sensitivity", "Radio", size=(20, 1), key="-HIGH-"),],
        [sg.Button("Show Corners", key="-SHOW-")],
    ]

    # GUI - define image viewer panel
    image_viewer_column = [
        [sg.Text("Image Viewer", size=(30, 2), font=("Open Sans Bold", 12), text_color='dark blue')],
        [sg.Text(size=(50, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    # GUI - define layout
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
        # Exit event - close window
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Event - choose a file
        elif event == "-FOLDER-":
            try:
                filename = values["-FOLDER-"]
                img = cv2.imread(filename)
                img_bytes = cv2.imencode(".png", img)[1].tobytes()
                window["-IMAGE-"].update(data=img_bytes)
            except:
                pass
        # Event - show corners button
        elif event == "-SHOW-":
            try:
                # Check which sensitivity radio is chosen
                if values["-LOW-"]:
                    th_params = low_sen_params
                elif values["-MEDIUM-"]:
                    th_params = medium_sen_params
                elif values["-HIGH-"]:
                    th_params = high_sen_params

                filename = values["-FOLDER-"]
                img = cv2.imread(filename)
                img_corners = detect_corners(img, **th_params)
                img_bytes = cv2.imencode(".png", img)[1].tobytes()
                window["-IMAGE-"].update(data=img_bytes)

            except Exception:
                sg.popup_error(f'AN EXCEPTION OCCURRED!\nPlease load a valid image!')

    window.close()