import warnings
from time import time

import cv2

from image_processor import ImageProcessor_Mediapipe
from self_modules.drawer import make_fig
from self_modules.udp_client import Udp_client


# for ignore numpy warning
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # setting
    wink_EAR_threshold = 12.5
    is_show_fig = True

    # prepare processor
    processor = ImageProcessor_Mediapipe( wink_EAR_threshold )
    # processor = Processor_InsightFace( wink_EAR_threshold )
    processor.prepare()

    # prepare udp_client
    udp_client = Udp_client()

    # start capture
    cap = cv2.VideoCapture(0)

    cnt = 0
    t0 = time()
    plt_data_dict = {
        "list": {
            "time": [],
            "left_EAR": [],
            "right_EAR": [],

        },
        "value": {
            "wink_EAR_threshold": wink_EAR_threshold
        },
    }
    while True:
        cnt += 1

        # read img
        ret, img_bgr = cap.read()
        t = time() - t0

        # flip img
        img_bgr = cv2.flip(img_bgr, 1)

        # detection, alignment, draw
        is_face_detected, is_wink_list, EAR_list = processor.run(img_bgr)
        print(f"L:{EAR_list[0]}, R:{EAR_list[1]}")

        # pass info to unity
        udp_client.send_msg_face_detected(is_face_detected)
        udp_client.send_msg_left_wink(is_wink_list[0])
        udp_client.send_msg_right_wink(is_wink_list[1])

        # show
        cv2.imshow('IMG: with FACE ALIGNMENT', img_bgr)

        if is_show_fig:
            # make fig images
            plt_data_dict["list"]["time"].append(t)
            plt_data_dict["list"]["left_EAR"].append(EAR_list[0])
            plt_data_dict["list"]["right_EAR"].append(EAR_list[1])
            if cnt > 100:
                for list_data in plt_data_dict["list"].values():
                    list_data.pop(0)
            fig_bgr = make_fig(plt_data_dict)

            # show
            cv2.imshow('FIG: TIME vs EAR', fig_bgr)

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
