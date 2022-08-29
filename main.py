import warnings

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.app.common import Face


# for ignore numpy warning
warnings.filterwarnings('ignore')


# functions
def draw_a_part_of_landmark_2d_68(img, landmark_2d_68, start_idx, end_idx, color, isClosed=False):
    points = landmark_2d_68[start_idx:end_idx + 1]
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, color, thickness=2, lineType=cv2.LINE_8)

def draw_landmark_2d_68(img, landmark_2d_68):
    normal_color = [0,255,0]
    left_color = [255, 0, 0]
    right_color = [0, 0, 255]
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 0, 16, normal_color)           # Jaw line
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 17, 21, normal_color)          # Left eyebrow
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 22, 26, normal_color)          # Right eyebrow
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 27, 30, normal_color)          # Nose bridge
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 30, 35, normal_color, True)    # Lower nose
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 36, 41, left_color, True)    # Left eye
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 42, 47, right_color, True)    # Right Eye
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 48, 59, normal_color, True)    # Outer lip
    draw_a_part_of_landmark_2d_68(img, landmark_2d_68, 60, 67, normal_color, True)    # Inner lip

# class
class Processor:
    def __init__(self, detector, aligner):
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

        # alignmert
        for face in faces:
            aligner.get(img, face)

        # draw
        for face in faces:
            landmark_2d_68 = face.landmark_3d_68[:,0:2]
            draw_landmark_2d_68(img, landmark_2d_68)


if __name__ == '__main__':
    # prepare detector, aligner
    detector = get_model('models/det_500m.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640))
    aligner = get_model('models/1k3d68.onnx')
    aligner.prepare(ctx_id=0)

    # prepare processor
    processor = Processor(detector, aligner)

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
