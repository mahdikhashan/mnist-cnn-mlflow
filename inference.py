import torch
assert torch.__version__ >= "2.0.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import mlflow
import mlflow.pytorch
assert mlflow.__version__ >= "1.0.0"
mlflow.set_tracking_uri("http://127.0.0.1:8083/")

import torchvision.transforms as transforms

from PIL import Image


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image_resized = transform(image)
    out = image_resized.view(-1, 1, 28, 28)
    return out

def predict(test_data):
    model_uri = "models:/mnist_cnn/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()

    inputs = preprocess_image(test_data["image"])

    with torch.no_grad():
        output = model(inputs)

    return output

if __name__ == "__main__":
    image_six = Image.open("inference_data/six-2.png")
    if image_six is None:
        print("Error: Image not loaded. Please check the file path.")
        exit(1)

    test_data = [{"image": image_six, "label": 6},]

    prediction = predict(test_data[0])
    predicted_digit = torch.argmax(prediction, dim=1).item()

    for i, input_data in enumerate(test_data):
        print(f"Input: {input_data['label']} -> Prediction: {predicted_digit}, "
              f"{'Correct' if (input_data['label'] == predicted_digit) else 'Failed' }")
