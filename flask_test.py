from flask import Flask
from flask import request
import torch
import torchvision.transforms as transforms
import utils.cnn_model as cnn_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
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

CORS(app, resources=r'/*')


@app.route("/get-query-code", methods=['POST'])
def get_query_code():
    a = request
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = transformations(image)
    image = torch.autograd.Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
    continuous_code = model(image).data.numpy().tolist()[0]
    discrete_code = (torch.sign(model(image).data) > 0).type(torch.LongTensor).numpy().tolist()[0]
    origincode = ''
    hashcode = ''
    for i in range(len(continuous_code)):
        if i == 0:
            origincode = origincode + str(continuous_code[i])
        else:
            origincode = origincode + ',' + str(continuous_code[i])
    for bit in discrete_code:
        hashcode = hashcode + str(bit)
    return {
        'hashcode': hashcode,
        'origincode': origincode
    };


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9001)
