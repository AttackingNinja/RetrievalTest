import os
from PIL import Image
import pandas as pd

image_infos = []
dir_path = "D:\\PyCharm-workspace\RetrievalTest\\result\\result-ADCH-Project-20-08-24-11-16-40\\9ZPDAVYEO42_IMG_8542"
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
