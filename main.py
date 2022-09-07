import warnings

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.app.common import Face


# for ignore numpy warning
warnings.filterwarnings('ignore')

# set wink threshold (EAR[%])
WINK_THRESHOLD = 23.0

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


# class
class Processor:
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

        if len(faces) != 0:
            face = faces[0]

            # alignmert
            aligner.get(img, face)
            landmark_2d_68 = face.landmark_3d_68[:, 0:2]

            # check wink
            start, end = LANDMARK_PARTS_DICT["Left Eye"]
            left_EAR = estimate_EAR( landmark_2d_68[start:end + 1] )
            left_wink = True if left_EAR <= WINK_THRESHOLD else False

            start, end = LANDMARK_PARTS_DICT["Right Eye"]
            right_EAR = estimate_EAR( landmark_2d_68[start:end + 1])
            right_wink = True if right_EAR <= WINK_THRESHOLD else False

            print(f"L:{left_EAR}, R:{left_EAR}")

            # draw
            draw_landmark_2d_68(img, landmark_2d_68)

            return left_wink, right_wink




if __name__ == '__main__':
    # prepare processor
    processor = Processor()
    processor.prepare()

    # start capture
    cap = cv2.VideoCapture(0)

    while True:
        # read img
        ret, img = cap.read()

        # flip img
        img = cv2.flip(img, 1)

        # detection, alignment, draw
        processor.run(img)

        # show
        cv2.imshow('wink_detection', img)

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
