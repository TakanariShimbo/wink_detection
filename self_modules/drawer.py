import cv2
import numpy as np


# functions
def draw_a_part_of_landmark_2d_68(img, landmark_2d_68, start_idx, end_idx, color, isClosed=False):
    points = landmark_2d_68[start_idx:end_idx + 1]
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, color, thickness=2, lineType=cv2.LINE_8)

def draw_landmark_2d_68(img, landmark_2d_68, LANDMARK_PARTS_DICT):
    # set color dict
    normal_color = [0,255,0]
    left_color = [255, 0, 0]
    right_color = [0, 0, 255]
    color_dict = {
        "Jaw Line": normal_color,
        "Left Eyebrow": normal_color,
        "Right Eyebrow": normal_color,
        "None Bridge": normal_color,
        "Lower Nose": normal_color,
        "Left Eye": left_color,
        "Right Eye": right_color,
        "Outer Lip": normal_color,
        "Inner Lip": normal_color
    }
    # set isClosed dict
    isClosed_dict = {
        "Jaw Line": False,
        "Left Eyebrow": False,
        "Right Eyebrow": False,
        "None Bridge": False,
        "Lower Nose": True,
        "Left Eye": True,
        "Right Eye": True,
        "Outer Lip": True,
        "Inner Lip": True
    }

    # draw each parts
    for key, (start, end) in LANDMARK_PARTS_DICT.items():
        draw_a_part_of_landmark_2d_68(img, landmark_2d_68, start, end, color_dict[key], isClosed_dict[key])
