import warnings

import cv2

from image_processor import ImageProcessor_Mediapipe
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

    while True:
        # read img
        ret, img = cap.read()

        # flip img
        img = cv2.flip(img, 1)

        # detection, alignment, draw
        is_face_detected, is_wink_list, EAR_list = processor.run(img)
        print(f"L:{EAR_list[0]}, R:{EAR_list[1]}")

        # pass info to unity
        udp_client.send_msg_face_detected(is_face_detected)
        udp_client.send_msg_left_wink(is_wink_list[0])
        udp_client.send_msg_right_wink(is_wink_list[1])

        # show
        cv2.imshow('wink_detection', img)

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
