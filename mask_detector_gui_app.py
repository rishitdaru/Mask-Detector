from tkinter import *
from keras.models import load_model
import cv2
import numpy as np

root = Tk()
root.title("Mask Detector")
root.iconbitmap("mask_icon.ico")
root.geometry("600x600")
root.configure(bg="black")

for i in range(3):
    for j in range(2):
        Grid.rowconfigure(root, i, weight=1)
        Grid.columnconfigure(root, j, weight=1)

stop_check = False


def start_btn():
    model = load_model('model-019.model')

    face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    source = cv2.VideoCapture(0)

    labels_dict = {0: 'MASK', 1: 'NO MASK'}
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

    while True:

        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Mask Detector", img)

        key = cv2.waitKey(1)
        if key == 32:
            cv2.waitKey()

        if key == 27 or stop_check:
            cv2.destroyAllWindows()
            source.release()
            break



def stop_btn():
    global stop_check
    stop_check = True


display_label = Label(root, text="The screen below detects whether a person has worn a mask or not", font=("Arial", 12))
display_label.grid(row=0, column=0, columnspan=2, sticky=E+W)

frame = LabelFrame(root, text="", padx=150, pady=150, bg="blue")
frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

start_img = PhotoImage(file="start.png").subsample(4, 4)
stop_img = PhotoImage(file="stop.png").subsample(4, 4)


start_btn = Button(root, image=start_img, borderwidth=0, bg="black", command=start_btn)
start_btn.grid(row=2, column=0, rowspan=2, sticky=E+W)

pause_label = Label(root, text="Press spacebar to pause", bg="black", fg="green", font=("Arial", 12))
pause_label.grid(row=2, column=1, sticky=E+W)

stop_label = Label(root, text="Press Esc to stop the video",  bg="black", fg="green", font=("Arial", 12))
stop_label.grid(row=3, column=1, sticky=E+W)

made_by_label = Label(root, text="Made by: Rishit Daru", anchor=E, bg="black", fg="red", font=("Arial", 12))
made_by_label.grid(row=4, column=1, columnspan=2, sticky=E+W, pady=20)


b = Label(frame, text="")
b.pack()

root.mainloop()


