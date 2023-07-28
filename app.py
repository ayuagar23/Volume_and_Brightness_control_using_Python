import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from math import hypot
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from flask import Flask, render_template, request, redirect,Response, session  # importing all the necessary libraries
import os
import hashlib
import numpy as np
import sqlite3
app = Flask(__name__)
app.secret_key = os.urandom(24)
conn = sqlite3.connect('minorprojdb.sqlite', check_same_thread=False)

cursor = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/saveuserdata', methods=['POST'])
def saveuserdata():
    Name = request.form.get('Name')
    Email = request.form.get('Email')
    Password = request.form.get('Password')
    cursor.execute( """INSERT INTO `user_data` (`Name`,`Email`,`Password`) VALUES ('{}','{}','{}')""".format(Name, Email, Password))
    conn.commit()

    return render_template('index.html')

@app.route('/signin', methods=['POST'])
def signin():
    Email= request.form.get('Emaill')
    Password = request.form.get('Passwordd')
    cursor.execute(
        """SELECT * FROM `user_data` WHERE `Email` LIKE '{}' AND `Password` LIKE '{}'""".format(Email, Password))
    users = cursor.fetchall()
    if len(users) > 0:
        return redirect('/about')
    else:
        return redirect('/signin')

@app.route('/about')
def about():
    return render_template('second_page.html')

@app.route('/video')
def video():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    volMin, volMax = volume.GetVolumeRange()[:2]

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if lmList != []:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [15, 220], [volMin, volMax])
            print(vol, length)
            volume.SetMasterVolumeLevel(vol, None)

            # Hand range 15 - 220
            # Volume range -63.5 - 0.0

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

@app.route('/video2')
def video2():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if lmList != []:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)

            bright = np.interp(length, [15, 220], [0, 100])
            print(bright, length)
            sbc.set_brightness(int(bright))

            # Hand range 15 - 220
            # Brightness range 0 - 100

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

# main driver function
if __name__ == '__main__':
    app.run(debug=True)