import os
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
import utils.cnn_model as cnn_model

image_infos = []
image_codes = []
dir_path = "data/Project/images"
model = cnn_model.CNNNet('resnet50', 48)
model.load_state_dict(torch.load('dict/adch-nuswide-48bits.pth', map_location=torch.device('cpu')))
model.eval()
Image.MAX_IMAGE_PIXELS = 1000000000
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
for img_name in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_name)
    img = Image.open(img_path)
    width = img.size[0]
    height = img.size[1]
    img_info = [img_name, width, height]
    image_infos.append(img_info)
data = image_infos
columns = ['image_name', 'width', 'height']
df = pd.DataFrame(columns=columns, data=data)
df.to_csv('image_info.csv', header=False, index=False)
for img_name in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_name)
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = transformations(img)
    img = torch.autograd.Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    continuous_code = model(img).data.numpy().tolist()[0]
    discrete_code = (torch.sign(model(img).data) > 0).type(torch.LongTensor).numpy().tolist()[0]
    origincode = ''
    hashcode = ''
    for i in range(len(continuous_code)):
        if i == 0:
            origincode = origincode + str(continuous_code[i])
        else:
            origincode = origincode + ' ' + str(continuous_code[i])
    for bit in discrete_code:
        hashcode = hashcode + str(bit)
    image_code = [hashcode, origincode]
    for i in range(6):
        sucode = hashcode[i * 8:i * 8 + 8]
        image_code.append(sucode)
    image_codes.append(image_code)
data = image_codes
columns = ['hashcode', 'origincode', 'subcode1', 'subcode2', 'subcode3', 'subcode4', 'subcode5', 'subcode6']
df = pd.DataFrame(columns=columns, data=data)
df.to_csv('image_code.csv', header=False, index=False)
