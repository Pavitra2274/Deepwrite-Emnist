import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import re
from torchvision import transforms

# ===== EMNIST Label Mapping =====
EMNIST_BYCLASS_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

# ===== CNN Model (same as training) =====
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 62)  # EMNIST ByClass
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# ===== Setup =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Pavitra Projects\Character Recognition\emnist_cnn.pth"

# Transform — matches EMNIST preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = CNNModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Flask app with correct template path
app = Flask(__name__, template_folder=r"C:\Pavitra Projects\Character Recognition\templates")

# ===== Helper: Convert base64 to PIL =====
def base64_to_pil(img_base64):
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    byte_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(byte_data)).convert('L')  # grayscale
    return image

# ===== Prediction =====
def predict_emnist(image):
    # Match EMNIST dataset orientation
    image = image.rotate(270, expand=True)  # -90 degrees
    image = ImageOps.mirror(image)  # flip horizontally

    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        pred_class = int(torch.argmax(outputs, dim=1).item())

    return EMNIST_BYCLASS_LABELS[pred_class], probs

# ===== Routes =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pil_image = base64_to_pil(data['image'])
    char, probs = predict_emnist(pil_image)
    return jsonify({
        'prediction_char': char,
        'confidence': round(float(max(probs)) * 100, 2),
        'probabilities': probs.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)

