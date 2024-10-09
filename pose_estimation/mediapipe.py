import cv2
import numpy as np
import mediapipe as mp
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

def upload_files():
    Tk().withdraw()
    file_paths = askopenfilenames(title="Select images", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    return file_paths

file_paths = upload_files()
images = {path: cv2.imread(path) for path in file_paths}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

for path, image in images.items():
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image_height, image_width, _ = image.shape
    if not results.pose_landmarks:
        continue
    
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )
    
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec
    )
    
    cv2.imshow(f'Annotated Image: {path}', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
