import warnings

import cv2
import numpy as np

from insightface.model_zoo import get_model
from insightface.app.common import Face
import mediapipe as mp

from udp_client import Udp_client


# for ignore numpy warning
warnings.filterwarnings('ignore')

# set wink threshold (EAR[%])
WINK_THRESHOLD = 20

# set
CORRESPONDENCE_LIST_68_468 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

# set landmark part dict
LANDMARK_PARTS_DICT = {
    "Jaw Line": [0, 16],
    "Left Eyebrow": [17, 21],
    "Right Eyebrow": [22, 26],
    "None Bridge": [27, 30],
    "Lower Nose": [30, 35],
    "Left Eye": [36, 41],
    "Right Eye": [42, 47],
    "Outer Lip": [48, 59],
    "Inner Lip": [60, 67]
}


# functions
def draw_a_part_of_landmark_2d_68(img, landmark_2d_68, start_idx, end_idx, color, isClosed=False):
    points = landmark_2d_68[start_idx:end_idx + 1]
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, color, thickness=2, lineType=cv2.LINE_8)

def draw_landmark_2d_68(img, landmark_2d_68):
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

def estimate_EAR(eye_landmark):
    p = eye_landmark
    vertical_elements = np.linalg.norm( p[1] - p[5] ) +np.linalg.norm( p[2] - p[4] )
    horizontal_elements = 2 * np.linalg.norm( p[0] - p[3] )
    percent_EAR = np.round(vertical_elements/horizontal_elements*100, 1)
    return percent_EAR

def check_wink( landmark_2d_68 ):
    # check wink
    start, end = LANDMARK_PARTS_DICT["Left Eye"]
    left_EAR = estimate_EAR(landmark_2d_68[start:end + 1])
    if left_EAR <= WINK_THRESHOLD:
        is_left_wink = True
    else:
        is_left_wink = False

    start, end = LANDMARK_PARTS_DICT["Right Eye"]
    right_EAR = estimate_EAR(landmark_2d_68[start:end + 1])
    if right_EAR <= WINK_THRESHOLD:
        is_right_wink = True
    else:
        is_right_wink = False

    print(f"L:{left_EAR}, R:{left_EAR}")
    return is_left_wink, is_right_wink


# class
class Processor_InsightFace:
    def __init__(self):
        self.detector = None
        self.aligner = None

    def prepare(self):
        # prepare detector, aligner
        detector = get_model('models/det_500m.onnx')
        detector.prepare(ctx_id=0, input_size=(640, 640))
        aligner = get_model('models/1k3d68.onnx')
        aligner.prepare(ctx_id=0)

        self.detector = detector
        self.aligner = aligner

    def run(self, img):
        detector = self.detector
        aligner = self.aligner

        # detection
        bboxes, kpss = detector.detect(img)
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            faces.append(face)

        is_face_detected = True if len(faces) != 0 else False
        is_left_wink = False
        is_right_wink = False
        if is_face_detected:
            face = faces[0]

            # alignment
            aligner.get(img, face)
            landmark_2d_68 = face.landmark_3d_68[:, 0:2]

            # check
            is_left_wink, is_right_wink = check_wink(landmark_2d_68)

            # draw
            draw_landmark_2d_68(img, landmark_2d_68)

        return is_face_detected, is_left_wink, is_right_wink


class Processor_Mediapipe:
    def __init__(self):
        self.detector = None
        self.aligner = None

    def prepare(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

        self.face_mesh = face_mesh

    def run(self, img):
        face_mesh = self.face_mesh

        # detection, alignment
        img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rbg)

        is_face_detected = True if results.multi_face_landmarks != None else False
        is_left_wink = False
        is_right_wink = False
        if is_face_detected:
            # convert landmark 468 to 68
            landmark_2d_68 = []
            landmark_2d_468 = results.multi_face_landmarks[0]
            height, width = img.shape[:2]
            for index in CORRESPONDENCE_LIST_68_468:
                x = landmark_2d_468.landmark[index].x * width
                y = landmark_2d_468.landmark[index].y * height
                landmark_2d_68.append([x, y])
            landmark_2d_68 = np.array(landmark_2d_68)

            # check
            is_left_wink, is_right_wink = check_wink(landmark_2d_68)

            # draw
            draw_landmark_2d_68(img, landmark_2d_68)

        return is_face_detected, is_left_wink, is_right_wink


if __name__ == '__main__':
    # prepare processor
    processor = Processor_Mediapipe()
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
        is_face_detected, is_left_wink, is_right_wink = processor.run(img)

        # pass info to unity
        udp_client.send_msg_face_detected(is_face_detected)
        udp_client.send_msg_left_wink(is_left_wink)
        udp_client.send_msg_right_wink(is_right_wink)

        # show
        cv2.imshow('wink_detection', img)

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
