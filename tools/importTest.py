import os
import cv2

print(os.getcwd())
weight_path = "face_eye_detection"
face_xml = 'haarcascade_frontalface_alt2.xml'
eye_glass_xml = 'haarcascade_eye_tree_eyeglasses.xml'

face_eye_dict = {
    "face_detector": cv2.CascadeClassifier(os.path.join(weight_path, face_xml)),
    "eye_detector": cv2.CascadeClassifier(os.path.join(weight_path, eye_glass_xml)),
}