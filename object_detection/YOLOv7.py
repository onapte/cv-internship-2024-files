import os
import subprocess
import glob
from roboflow import Roboflow
import requests
from PIL import Image

subprocess.run(["git", "clone", "https://github.com/SkalskiP/yolov7.git"])

os.chdir("yolov7")

subprocess.run(["git", "checkout", "fix/problems_associated_with_the_latest_versions_of_pytorch_and_numpy"])

subprocess.run(["pip", "install", "-r", "requirements.txt"])

subprocess.run(["pip", "install", "roboflow"])

rf = Roboflow(api_key="YOUR API KEY")
project = rf.workspace("YOUR-WORKSPACE").project("YOUR-PROJECT")
dataset = project.version(1).download("yolov7")

os.chdir("/content/yolov7")
subprocess.run(["wget", "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt"])

subprocess.run([
    "python", "train.py", "--batch", "16", "--epochs", "55",
    "--data", f"{dataset.location}/data.yaml", "--weights", "yolov7_training.pt", "--device", "0"
])

subprocess.run([
    "python", "detect.py", "--weights", "runs/train/exp/weights/best.pt", "--conf", "0.1", 
    "--source", f"{dataset.location}/test/images"
])

i = 0
limit = 10000
for imageName in glob.glob('runs/detect/exp/*.jpg'):
    if i < limit:
        img = Image.open(imageName)
        img.show()
        print("\n")
    i += 1

rf = Roboflow(api_key="<MY API KEY>")
project = rf.workspace("workspace").project("project")
dataset = project.version(1)

project.version(dataset.version).deploy(model_type="yolov7", model_path=f"runs/train/exp/weights/")

infer_payload = {
    "image": {
        "type": "url",
        "value": "image_url",
    },
    "confidence": 0.75,
    "iou_threshold": 0.5,
    "api_key": "api_key",
}

res = requests.post(
    f"http://localhost:9001/{workspace_id}/{model_id}",
    json=infer_payload
)

predictions = res.json()
