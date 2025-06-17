# inference.py

# Make the decision of 


from PIL import Image
import torch

# predict_image
# 
def predict_image(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img)
        predicted = torch.argmax(output, 1)
        return 'day' if predicted.item() == 0 else 'night'
