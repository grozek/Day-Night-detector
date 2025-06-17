from fastapi import FastAPI, File, UploadFile
from model import CNN
from torchvision import transforms
from PIL import Image
import torch
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = CNN()
model.load_state_dict(torch.load("day_night_model.pth", map_location="mps"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = (torch.sigmoid(output) > 0.5).int()
    return {"result": "Day" if pred == 0 else "Night"}
