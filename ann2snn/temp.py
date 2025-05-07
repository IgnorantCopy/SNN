import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from PIL import Image


dataset_root = "/home/nju-student/mkh/datasets"
path = os.path.join(dataset_root, "flowers")
labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
          "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
          "tulip", "water_lily"]
data = []
for label in labels:
    img_dir = os.path.join(path, label)
    image_names = os.listdir(img_dir)
    for image_name in image_names:
        img_path = os.path.join(img_dir, image_name)
        data.append({"image_path": img_path, "label": label})
data = pd.DataFrame(data)

num_of_images = []
for i, label in enumerate(labels):
    df = data[data["label"] == label]
    num_of_images.append(len(df))
fig = go.Figure(data=go.Bar(x=labels, y=num_of_images))
fig.update_layout(title={
    "text": "Data Distribution",
    "xanchor": "center",
    "yanchor": "top",
    "x": 0.5,
    "y": 0.9
}, xaxis_title="Label", yaxis_title="Number of Images")
fig.show()