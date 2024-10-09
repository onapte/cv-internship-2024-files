import os
import subprocess
import patoolib

subprocess.run(["git", "clone", "https://github.com/lannguyen0910/deep-efficient-person-reid"])
os.chdir("/content/deep-efficient-person-reid")

subprocess.run(["pip", "install", "-r", "requirements.txt"])
subprocess.run(["pip", "install", "-U", "PyYAML"])
os.chdir("/content/deep-efficient-person-reid/dertorch/")

subprocess.run(["gdown", "--id", "12l1Z8qgVoLpjewwWwC1AN7_bK1_SIV_w"])
subprocess.run(["gdown", "https://drive.google.com/uc?id=12l1Z8qgVoLpjewwWwC1AN7_bK1_SIV_w"])
subprocess.run(["gdown", "https://drive.google.com/uc?id=1gscms5dTajTmJ-0DrqOVTyjPZCi20Ku3"])
subprocess.run(["gdown", "https://drive.google.com/uc?id=1hV9di9QRlKH1rMeqjlgc3HrpdsHuJKat"])
subprocess.run(["gdown", "https://drive.google.com/uc?id=1Obd2jyg753Hiil86J0np51IQVWGp7AZf"])

subprocess.run(["pip", "install", "patool"])

patoolib.extract_archive("cuhk03_release.rar", outdir="./")

subprocess.run(["unzip", "weights_efficientnetv2_market-20210704T012957Z-001.zip"])
subprocess.run(["unzip", "weights_efficientnetv2_cuhk-20210704T014847Z-001.zip"])
subprocess.run(["unzip", "weights-resnet50-market-20210704T021220Z-001.zip"])
subprocess.run(["unzip", "weights-resnet50-cuhk-20210704T023616Z-001.zip"])

subprocess.run(["gdown", "https://drive.google.com/uc?id=1rTYuStYA64U1qb7uC-h8DOhv99Ztxaxz"])
subprocess.run(["gdown", "https://drive.google.com/uc?id=162_RPNfOguXy21nrYZZSvEtex9ombYtq"])

subprocess.run(["python", "train.py", "--config_file=efficientnetv2_market"])
subprocess.run(["python", "test.py", "--config_file=efficientnetv2_market"])
subprocess.run(["python", "visualize.py", "--config_file=efficientnetv2_market"])
