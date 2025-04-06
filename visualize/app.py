from flask import Flask, request
import base64
import os
from flask_cors import CORS
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from spikingjelly.activation_based import ann2snn
import numpy as np
import PIL.Image as Image
import PIL.ImageOps as ImageOps
from ann2snn.net import ConvNet


model_path = "../ann2snn/models/ann_MNIST_256_Adam_1e-02.pth"
dataset_root = "E:/DataSets"
batch_size = int(model_path.split('_')[2])
mode = "99.9%"
T = 50
image_size = 28
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

ann_model = ConvNet(10, image_size, batch_size, num_channels=1)
ann_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
converter = ann2snn.Converter(mode=mode, dataloader=train_loader)
snn_model = converter(ann_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
snn_model.to(device)


def preprocess(filename):
    image = Image.open(filename)
    alpha = image.split()[3]
    bg_mask = alpha.point(lambda x: 255 - x)
    image = image.convert('RGB')
    image.paste((255, 255, 255), None, bg_mask)
    image = image.resize((image_size, image_size))
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = np.array(image)
    image = image.reshape(1, image_size, image_size)
    return image


def predict(filename):
    image = preprocess(filename)
    image = transform_test(image)
    image = image.permute(1, 2, 0)
    image = image.unsqueeze(0)
    snn_model.eval()
    with torch.no_grad():
        image = image.to(device)
        for m in snn_model.modules():
            if hasattr(m, 'reset'):
                m.reset()
        for t in range(T):
            if t == 0:
                out = snn_model(image)
            else:
                out += snn_model(image)
        # out = ann_model(image)
    return torch.argmax(out, dim=1).item()

app = Flask(__name__)
CORS(app)

SAVE_DIR = 'images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    try:
        data = request.json.get('data')
        if not data:
            return {'message': 'No data provided'}, 400

        if data.startswith('data:image/png;base64,'):
            data = data.split(',')[1]

        image_data = base64.b64decode(data)
        file_name = os.path.join(SAVE_DIR, 'image.png')
        with open(file_name, 'wb') as f:
            f.write(image_data)

        label = predict(file_name)

        return {'message': str(label), 'file_path': file_name}, 200
    except Exception as e:
        return {'message': f'Error: {str(e)}'}, 500


if __name__ == '__main__':
    app.run(host='10.54.36.75', port=5000)