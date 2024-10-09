import cv2
import matplotlib.pyplot as plt
import os

!git clone https://github.com/AlexeyAB/darknet
!wget https://pjreddie.com/media/files/yolov3.weights

def imShow(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()

def upload():
    print("Use a file manager to upload files to the system")

def download(path):
    print(f"File can be downloaded from {path}")

!./darknet detect cfg/yolov3.cfg yolov3.weights data/person.jpg
imShow('predictions.jpg')

!./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
imShow('predictions.jpg')

download('cfg/yolov3.cfg')

!cp /path/to/your/generate_train.py ./

!python generate_train.py

!wget http://pjreddie.com/media/files/darknet53.conv.74

!./darknet detector train data/obj.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show

%cd cfg
!sed -i 's/batch=64/batch=1/' yolov3_custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov3_custom.cfg
%cd ..

!./darknet detector test data/obj.data cfg/yolov3_custom.cfg /path/to/your/yolov3_custom_last.weights /path/to/your/safari.jpg -thresh 0.3
imShow('predictions.jpg')
