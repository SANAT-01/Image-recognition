import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 90, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(90, 100, 5)
        self.fc1 = nn.Linear(100 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.load_state_dict(torch.load(r'D:\DOC\COLLEGE DOC\NOTES\SEMESTER 4\DS252 LAB\cifar_net.pth'))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def pre_image(image_path,model):
    img = Image.fromarray(np.uint8(image_path)).convert('RGB')
    transform_norm = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    with torch.no_grad():
        model.eval()
        output =model(img_normalized)
        _, predicted = torch.max(output, 1)
        class_name = classes[predicted]
        return class_name

cap = cv2.VideoCapture(0)
frame_rate = 70
prev = 0

while (cap.isOpened()):
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        predict_class = pre_image(frame,net)
        cv2.imshow('frame', frame)
        print(predict_class)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()