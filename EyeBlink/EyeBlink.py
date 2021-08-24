
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import tkinter as tk
import threading
        
morse_dict = {}

quitReq = False


class App(threading.Thread):

    morsLabel = None
    text_box = None
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        #f = open(date, 'w')
        #f.write(self.get('1.0', 'end'))

        global quitReq
        quitReq = True
        morsLabel = None
        text_box = None
        self.root.quit()

    def updateMorsLabel(self,MORS):
        if self.morsLabel != None:
            self.morsLabel['text'] = MORS

    def addtoTextBox(self,txt):
        if self.text_box != None:
            if self.text_box != txt:
                self.text_box.insert(tk.END, txt)

    def deletefromTextBox(self):
        if self.text_box != None:
            ln=len(self.text_box.get("1.0",tk.END))
            if ln > 1:
                chr = "1."+ str(ln-2)
                self.text_box.delete(chr)

    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        master = self.root

        r = 0
        c = 0
        for key in morse_dict:
            l1=tk.Label(master, text = morse_dict[key]+" "+key, font=("Arial", 24))
            l1.grid(row = r, column = c, sticky = tk.W, pady = 2)
            c += 1
            if c == 10:
                c = 0
                r += 1
        r += 1
        self.morsLabel=tk.Label(master, text = " ", font=("Arial", 24))
        self.morsLabel.grid(row = r, column = 5, sticky = tk.W, pady = 2)

        r += 1
        self.text_box = tk.Text( font=("Arial", 24), height=8)
        self.text_box.grid(row = r, column = 0, sticky = tk.W, pady = 2,columnspan = 10)

        self.root.mainloop()



def create_mors_dict():
    global morse_dict
    morse_dict['.-'] = 'A'
    morse_dict['-...'] = 'B'
    morse_dict['-.-.'] = 'C'
    morse_dict['-..'] = 'D'
    morse_dict['.'] = 'E'
    morse_dict['..-.'] = 'F'
    morse_dict['--.'] = 'G'
    morse_dict['....'] = 'H'
    morse_dict['..'] = 'I'
    morse_dict['.---'] = 'J'
    morse_dict['-.-'] = 'K'
    morse_dict['.-..'] = 'L'
    morse_dict['--'] = 'M'
    morse_dict['-.'] = 'N'
    morse_dict['---'] = 'O'
    morse_dict['.--.'] = 'P'
    morse_dict['--.-'] = 'Q'
    morse_dict['.-.'] = 'R'
    morse_dict['...'] = 'S'
    morse_dict['-'] = 'T'
    morse_dict['..-'] = 'U'
    morse_dict['...-'] = 'V'
    morse_dict['-.--'] = 'Y'
    morse_dict['--..'] = 'Z'
    morse_dict[' '] = ''
    morse_dict['.-.-'] = 'Sil'
    morse_dict['  '] = ''
    morse_dict['-..-'] = 'Boş'
    morse_dict['   '] = ''
    morse_dict['    '] = ''
    morse_dict['-----'] = '0'
    morse_dict['.----'] = '1'
    morse_dict['..---'] = '2'
    morse_dict['...--'] = '3'
    morse_dict['....-'] = '4'
    morse_dict['.....'] = '5'
    morse_dict['-....'] = '6'
    morse_dict['--...'] = '7'
    morse_dict['---..'] = '8'
    morse_dict['----.'] = '9'
    morse_dict['.-.-.-'] = '.'
    morse_dict['--..--'] = ','
    morse_dict['..--..'] = '?'
    morse_dict['-.-.--'] = '!'
    morse_dict['-..-.'] = '/'
    morse_dict['-.--.'] = '('
    morse_dict['-.--.-'] = ')'
    morse_dict['.-...'] = '&'
    morse_dict['---...'] = ':'
    morse_dict['-.-.-.'] = ';'
    morse_dict['-...-'] = '='
    morse_dict['.-.-.'] = '+'
    morse_dict['-....-'] = '-'
    morse_dict['..--.-'] = '_'
    morse_dict['.-..-.'] = '"'
    morse_dict['...-..-'] = '$'
    morse_dict['.--.-.'] = '@'
    morse_dict['..-.-'] = '¿'
    morse_dict['--...-'] = '¡'
 



def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

create_mors_dict()
app = App()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", required=False,
    help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.20
EYE_AR_MIN= 0.2
EYE_AR_MAX= 0.2

EYE_AR_CONSEC_FRAMES = 4

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
OLD_SYM = 0
SYM = 0
MORS =''
TXT =''
WCOUNTER = 0
DECISION_THRESHOLD = 30
WAITC = 300

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
if args["shape_predictor"] is not None:
    predictor = dlib.shape_predictor(args["shape_predictor"])
else:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
#time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # calculate/update threshold
        if ear > EYE_AR_MAX:
            EYE_AR_MAX = ear * 1.1
        
        EYE_AR_THRESH = EYE_AR_THRESH * 0.995 + EYE_AR_MAX * 0.004
        #EYE_AR_THRESH = EYE_AR_THRESH * 0.99 + EYE_AR_MAX * 0.008
        EYE_AR_MAX = EYE_AR_MAX * 0.995
        if ear < EYE_AR_MIN:
            EYE_AR_MIN = ear
        EYE_AR_MIN = EYE_AR_MIN * 1.02

        WAITC -=1
        if  WAITC<0:
            WAITC = 0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if WAITC == 0:
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                WCOUNTER = 0

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        WAITC = 8
                        if COUNTER <=10:
                            SYM ='.'
                        else:          
                            SYM ='-'
                else:
                    SYM =''
                # reset the eye frame counter
                COUNTER = 0
                WCOUNTER += 1


        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MORS: {}".format(MORS), (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "TRESH: {}".format(EYE_AR_THRESH), (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAX: {}".format(WAITC), (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        if MORS in morse_dict:
            app.updateMorsLabel(MORS+" "+morse_dict[MORS])
        else:
            app.updateMorsLabel(MORS)

        if SYM != OLD_SYM:
            OLD_SYM = SYM
            if SYM != '':
                MORS +=str(SYM)


    if WCOUNTER > DECISION_THRESHOLD:
        if MORS in morse_dict:
            if morse_dict[MORS] =='Sil':
                app.deletefromTextBox()
            elif morse_dict[MORS] =='Boş':
                app.addtoTextBox(" ")
            else:
                app.addtoTextBox(morse_dict[MORS])
        MORS =''

    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q") or quitReq:
        break

# do a bit of cleanup
app.callback()
cv2.destroyAllWindows()
vs.stop()
