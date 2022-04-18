import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound
import face_recognition
import sys
import os



cap = cv2.VideoCapture(0)
img1 = face_recognition.load_image_file('Sandesh.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
face =face_recognition.face_locations(img1)[0]
encodeFace = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), 5)
print("1")

while True:
    ret,img = cap.read()

    bbox, label, conf = cv.detect_common_objects(img)
    output = draw_bbox(img,bbox,label,conf)
    out=label
    print("2")
    print(out)
    check = 'person'
    if check in out:
        print("3")
        frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = frame1[:, :, ::-1]
        try:
            print('4')
            faceTest = face_recognition.face_locations(frame)[0]
            encodefaceTest = face_recognition.face_encodings(frame)[0]
            cv2.rectangle(frame1, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (0, 255, 0), 5)
            result = face_recognition.compare_faces([encodeFace], encodefaceTest)
            face_distance = face_recognition.face_distance([encodeFace], encodefaceTest)
            print(result)
            if (result[0] == True):
                print('5')
                cv2.putText(frame1, "Sandesh", (faceTest[3], faceTest[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.remove("person")
                out.insert(0,"Sandesh")
                print(out)

            else:
                print('6')
                pass
                print(out)
        except IndexError as e:
            print('7')
            print(e)
        def convert(text):
            print("8")
            audio = gTTS(text)
            audio.save('output_face.mp3')
            playsound('output_face.mp3')
            os.remove('output_face.mp3')
        convert('Your list contains: {}'.format(out))
        print(out)
        cv2.imshow("test", frame1)
    else:
        print("9")
        def convert(text):
            print("10")
            audio = gTTS(text)
            audio.save('output.mp3')
            playsound('output.mp3')
            os.remove('output.mp3')
        convert('Your list contains: {}'.format(out))
        print(out)
        cv2.imshow("test", img)
    k=cv2.waitKey(1)
    if k ==27 & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()