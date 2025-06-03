from fastapi import FastAPI, File, UploadFile
from model import CNN
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Load model
model = CNN()
model.load_state_dict(torch.load("day_night_model.pth", map_location="mps"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match model input
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()
    return {"result": "day" if prediction == 0 else "night"}
