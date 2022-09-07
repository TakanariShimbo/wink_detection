import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def make_fig(plt_data_dict):
    # make fig
    fig, ax = plt.subplots()

    # draw line
    ax.plot(plt_data_dict["time"], plt_data_dict["left_EAR"], 'b,-')
    ax.plot(plt_data_dict["time"], plt_data_dict["right_EAR"], 'r,-')

    # draw dot
    t = plt_data_dict["time"][-1]
    left_EAR = plt_data_dict["left_EAR"][-1]
    right_EAR = plt_data_dict["right_EAR"][-1]
    ax.scatter([t], [left_EAR], c='b')
    ax.scatter([t], [right_EAR], c='r')

    # set lim
    shift = 0.5
    ax.set_xlim([t + shift - 5.0, t + shift])
    ax.set_ylim([0, 40])

    # convert fig -> img
    fig.canvas.draw()
    plt.close()
    img_fig_rgba = np.array(fig.canvas.renderer.buffer_rgba())
    img_fig_bgr = cv2.cvtColor(img_fig_rgba, cv2.COLOR_RGBA2BGR)

    return img_fig_bgr