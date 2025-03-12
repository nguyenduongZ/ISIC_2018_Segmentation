import os, sys
import numpy as np
import cv2
import albumentations as A

from shutil import copyfile
from torch.utils.data import random_split

sys.path.append(os.path.abspath('./'))
from utils import helpers as h

_root = "/".join(__file__.split("/")[:-1]) + "/source/lesion"

TRAIN_INPUT_FOLDER = os.path.join(_root, 'ISIC2018_Task1-2_Training_Input')
VALID_INPUT_FOLDER = os.path.join(_root, 'ISIC2018_Task1-2_Validation_Input')
TEST_INPUT_FOLDER = os.path.join(_root, 'ISIC2018_Task1-2_Test_Input')

TRAIN_GT_FOLDER = os.path.join(_root, 'ISIC2018_Task1_Training_GroundTruth')
VALID_GT_FOLDER = os.path.join(_root, 'ISIC2018_Task1_Validation_GroundTruth')
TEST_GT_FOLDER = os.path.join(_root, 'ISIC2018_Task1_Test_GroundTruth')

def get_files(folder):
    files = h.listdir(folder)
    files.sort()
    files = [f for f in files if not ".txt" in f]
    files = [os.path.join(folder, f) for f in files]

    return files

train_input = get_files(TRAIN_INPUT_FOLDER)
valid_input = get_files(VALID_INPUT_FOLDER)
test_input = get_files(TEST_INPUT_FOLDER)

train_gt = get_files(TRAIN_GT_FOLDER)
valid_gt = get_files(VALID_GT_FOLDER)
test_gt = get_files(TEST_GT_FOLDER)

inputs = train_input + valid_input + test_input
gts = train_gt + valid_gt + test_gt

train_valid_test_split = (0.8, 0.1, 0.1)

test_count = int(train_valid_test_split[2] * len(inputs))
valid_count = test_count
train_count = len(inputs) - test_count * 2

print(train_count, valid_count, test_count)
assert(train_count + valid_count + test_count == len(inputs))

all_files = np.array(list(zip(inputs, gts)))
np.random.seed(42)
np.random.shuffle(all_files)

train_files = all_files[: train_count]
valid_files = all_files[train_count:train_count+valid_count]
test_files = all_files[-test_count:]

def save_files(files, folder):
    h.mkdir(folder)
    h.mkdir(os.path.join(folder, "input"))
    h.mkdir(os.path.join(folder, "label"))

    for input_file, gt_file in files:
        file_name = input_file.split('/')[-1]
        input_destination = os.path.join(folder, 'input', file_name)
        gt_destination = os.path.join(folder, "label", file_name.replace(".jpg", ".png"))

        resized = A.Compose([
            A.Resize(384, 512)
        ])

        input_img = cv2.imread(input_file)
        input_img = resized(image=input_img)["image"]

        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        gt_img = resized(image=gt_img)['image']

        cv2.imwrite(input_destination, input_img)
        cv2.imwrite(gt_destination, gt_img)

save_files(train_files, _root + "/train")
save_files(valid_files, _root + "/valid")
save_files(test_files, _root + "/test")