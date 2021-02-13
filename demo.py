import argparse
import os, sys
import json
import cv2
from facenet_pytorch import MTCNN
from utils.utils import *
import time

#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
args = parser.parse_args()
#################################################################

with open('./label.json', 'r') as json_file:
    label = json.load(json_file)

class Demo(object):
    def __init__(self, data):
        self.data = data
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(self.device)
        self.label = label
        self.gender_model = def_model('gender', self.device)
        self.gaze_model = def_model('gaze', self.device)
        self.emotion_model = def_model('emotion', self.device)
        self.multimodal_model = def_model('multimodal', self.device)

    def run(self, data):
        # video capture using data
        cap = cv2.VideoCapture(data)

        while True:
            start = time.time()
            # capture image from camera
            ret, frame = cap.read()

            try:
                # detect face box and probability
                boxes, probs = self.mtcnn.detect(frame, landmarks=False)

                # draw box on frame
                frame = draw_bbox(frame, boxes, probs)

                # perform only when face is detected
                if len(boxes) > 0:

                    # extract the face roi
                    rois = detect_rois(boxes)

                    for roi in rois:
                        (start_Y, end_Y, start_X, end_X) = roi
                        face = frame[start_Y:end_Y, start_X:end_X]
                        print('detect time: ', time.time()-start)
                        pred_time = time.time()
                        # run the classifier on bounding box
                        gender_i = predict(face, self.gender_model, self.device)
                        gaze_i = predict(face, self.gaze_model, self.device)
                        emotion_i = predict(face, self.emotion_model, self.device)
                        multimodal_i = predict(face, self.multimodal_model, self.device)

                        # assign labeling
                        cv2.putText(frame, label['gender'][gender_i], (end_X-50, start_Y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label['gaze'][gaze_i], (end_X-50, start_Y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label['emotion'][emotion_i], (end_X-50, start_Y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label['multimodal'][multimodal_i], (end_X-50, start_Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        print('predict time: ', time.time()-pred_time)
            except:
                pass

            # show the frame
            cv2.imshow('Demo', frame)
            
            # save image
            # cv2.imwrite('sample/sample.jpg', window)

            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Interrupted by user!')
                break
        
        # clear program and close windows
        cap.release()
        cv2.destroyAllWindows()
        print('All done!')

if args.image:
    if not os.path.isfile(args.image):
        print("Input image file {} doesn't exist".format(args.image))
        sys.exit(1)
    fcd = Demo(args.image)
    fcd.run(args.image)
elif args.video:
    if not os.path.isfile(args.video):
        print("Input video file {} dosen't exist".format(args.video))
        sys.exit(1)
    os.system('ffmpeg -i {} -ar 48000 -f wav {}.wav'.format(args.video, args.video[:args.video.find('.')]))
    fcd = Demo(args.video)
    fcd.run(args.video)
else:
    fcd = Demo(args.image)
    fcd.run(args.src)
