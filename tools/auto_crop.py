import os
import cv2
import sys
import dlib
from imutils import face_utils

def crop_boundary(top, bottom, left, right, faces, image):
    h, w = image.shape
    if faces:
        top = 0
        left = max(0, left - int(w/5))
        right = min(w, right + int(w/5))
        bottom = min(h, bottom + int(h/13))
#     else:
#         top = 0
#         left = max(0, left - 50)
#         right += 50
#         bottom += 50

    return (top, bottom, left, right)

def crop_face(imgpath, roi):
    xywh_list = []
    frame = cv2.imread(imgpath)
    basename = os.path.basename(imgpath)
    basename_without_ext = os.path.splitext(basename)[0]
    if frame is None:
        print(f"Invalid file path: [{imgpath}]")
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x1, y1, w, h = roi
    intbox = list(map(int, (x1, y1, x1 + w, y1 + h)))
    
    intbox[0] = max(0, intbox[0])
    intbox[1] = max(0, intbox[1])
    intbox[2] = min(frame.shape[1], intbox[2])
    intbox[3] = min(frame.shape[0], intbox[3])
    
    if intbox[1] == intbox[3] or intbox[0] == intbox[2]:
        print(f"There is no pedestrian: [{intbox[1], intbox[3], intbox[0], intbox[2]}]")
        return None
    frame = frame[intbox[1]:intbox[3], intbox[0]:intbox[2]]
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print(f"There is no pedestrian: [{intbox[1], intbox[3], intbox[0], intbox[2]}]")
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    if not len(rects):
        print(f"Sorry. HOG could not detect any faces from your image.\n[{imgpath}]")
        return None
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        top, bottom, left, right = crop_boundary(y, y + h, x, x + w, len(rects) <= 2, gray)
        face_roi = [intbox[0] + left, intbox[1] + top, intbox[0] + left + right, intbox[1] + top + bottom]
        xywh_list.append(face_roi)
        # crop_img_path = os.path.join(savePath, f"{basename_without_ext}_crop_{i}.png")
        # crop_img = frame[top: bottom, left: right]
        # cv2.imwrite(crop_img_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    print(f"SUCCESS: [{basename}]")
    return xywh_list