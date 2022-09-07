import warnings
from time import time

import cv2

from image_processor import ImageProcessor_Mediapipe
from self_modules.drawer import make_fig
from self_modules.udp_client import Udp_client


# for ignore numpy warning
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # prepare processor
    processor = ImageProcessor_Mediapipe()
    # processor = Processor_InsightFace()
    processor.prepare()

    # prepare udp_client
    udp_client = Udp_client()

    # start capture
    cap = cv2.VideoCapture(0)

    cnt = 0
    t0 = time()
    plt_data_dict = {
        "time": [],
        "left_EAR":[],
        "right_EAR":[]
    }
    while True:
        cnt += 1

        # read img
        ret, img = cap.read()
        t = time() - t0

        # flip img
        img = cv2.flip(img, 1)

        # detection, alignment, draw
        is_face_detected, is_wink_list, EAR_list = processor.run(img)
        print(f"L:{EAR_list[0]}, R:{EAR_list[1]}")

        # pass info to unity
        udp_client.send_msg_face_detected(is_face_detected)
        udp_client.send_msg_left_wink(is_wink_list[0])
        udp_client.send_msg_right_wink(is_wink_list[1])

        # make fig images
        plt_data_dict["time"].append(t)
        plt_data_dict["left_EAR"].append(EAR_list[0])
        plt_data_dict["right_EAR"].append(EAR_list[1])
        if cnt > 100:
            for data_list in plt_data_dict.values():
                data_list.pop(0)
        img_fig_bgr = make_fig( plt_data_dict )


        # show
        cv2.imshow('wink_detection', img)
        cv2.imshow('ear_fig', img_fig_bgr)

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
