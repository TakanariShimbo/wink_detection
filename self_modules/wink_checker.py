import numpy as np

# functions
def estimate_EAR(eye_landmark):
    p = eye_landmark
    vertical_elements = np.linalg.norm( p[1] - p[5] ) +np.linalg.norm( p[2] - p[4] )
    horizontal_elements = 2 * np.linalg.norm( p[0] - p[3] )
    percent_EAR = np.round(vertical_elements/horizontal_elements*100, 1)
    return percent_EAR

def check_wink( landmark_2d_68, wink_threshold, LANDMARK_PARTS_DICT ):
    # check wink
    start, end = LANDMARK_PARTS_DICT["Left Eye"]
    left_EAR = estimate_EAR(landmark_2d_68[start:end + 1])
    if left_EAR <= wink_threshold:
        is_left_wink = True
    else:
        is_left_wink = False

    start, end = LANDMARK_PARTS_DICT["Right Eye"]
    right_EAR = estimate_EAR(landmark_2d_68[start:end + 1])
    if right_EAR <= wink_threshold:
        is_right_wink = True
    else:
        is_right_wink = False

    is_wink_list = [is_left_wink, is_right_wink]
    EAR_list = [left_EAR, right_EAR]
    return is_wink_list, EAR_list