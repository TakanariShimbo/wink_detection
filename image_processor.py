import warnings

import cv2
import numpy as np

from insightface.model_zoo import get_model
from insightface.app.common import Face
import mediapipe as mp

from self_modules.wink_checker import check_wink
from self_modules.drawer import draw_landmark_2d_68


# for ignore numpy warning
warnings.filterwarnings('ignore')


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

# set
CORRESPONDENCE_LIST_68_468 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


# classes
class ImageProcessor_Mediapipe:
    def __init__(self, wink_EAR_threshold):
        self.face_mesh = None
        self.wink_EAR_threshold = wink_EAR_threshold


    def prepare(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

        self.face_mesh = face_mesh

    def run(self, img_bgr):
        face_mesh = self.face_mesh
        wink_EAR_threshold = self.wink_EAR_threshold

        # detection, alignment
        img_rbg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rbg)

        is_face_detected = True if results.multi_face_landmarks != None else False
        if is_face_detected:
            # convert landmark 468 to 68
            landmark_2d_68 = []
            landmark_2d_468 = results.multi_face_landmarks[0]
            height, width = img_bgr.shape[:2]
            for index in CORRESPONDENCE_LIST_68_468:
                x = landmark_2d_468.landmark[index].x * width
                y = landmark_2d_468.landmark[index].y * height
                landmark_2d_68.append([x, y])
            landmark_2d_68 = np.array(landmark_2d_68)

            # check
            is_wink_list, EAR_list = check_wink(landmark_2d_68, wink_EAR_threshold, LANDMARK_PARTS_DICT)

            # draw
            draw_landmark_2d_68(img_bgr, landmark_2d_68, LANDMARK_PARTS_DICT)

        else:
            is_wink_list = [False, False]
            EAR_list = [np.nan, np.nan]

        return is_face_detected, is_wink_list, EAR_list


class ImageProcessor_InsightFace:
    def __init__(self, wink_EAR_threshold):
        self.detector = None
        self.aligner = None
        self.wink_EAR_threshold = wink_EAR_threshold

    def prepare(self):
        # prepare detector, aligner
        detector = get_model('models/det_500m.onnx')
        detector.prepare(ctx_id=0, input_size=(640, 640))
        aligner = get_model('models/1k3d68.onnx')
        aligner.prepare(ctx_id=0)

        self.detector = detector
        self.aligner = aligner

    def run(self, img_bgr):
        detector = self.detector
        aligner = self.aligner
        wink_EAR_threshold = self.wink_EAR_threshold

        # detection
        bboxes, kpss = detector.detect(img_bgr)
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            faces.append(face)

        is_face_detected = True if len(faces) != 0 else False
        if is_face_detected:
            face = faces[0]

            # alignment
            aligner.get(img_bgr, face)
            landmark_2d_68 = face.landmark_3d_68[:, 0:2]

            # check
            is_wink_list, EAR_list = check_wink(landmark_2d_68, wink_EAR_threshold, LANDMARK_PARTS_DICT)

            # draw
            draw_landmark_2d_68(img_bgr, landmark_2d_68, LANDMARK_PARTS_DICT)
        else:
            is_wink_list = [False, False]
            EAR_list = [np.nan, np.nan]

        return is_face_detected, is_wink_list, EAR_list