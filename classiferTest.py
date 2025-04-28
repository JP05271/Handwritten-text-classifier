import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# Load saved weights
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Transform for image, changes to grayscale and resizes to correct dimentions
transform = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# loads and transforms image
image_path = 'Image of 3.png'
image = Image.open(image_path).convert('L')  # Convert to grayscale

# Inverts colors
image = ImageOps.invert(image)

# converts to pure black/white
image = image.point(lambda x: 0 if x < 128 else 255, '1')

# Find bounding box of the digit
bbox = image.getbbox()
if bbox:
    image = image.crop(bbox)

image = image.resize((20, 20), Image.Resampling.LANCZOS)

# Creates a 28x28 image and places 20x20 image in the center
new_image = Image.new('L', (28, 28), 0)
upper_left = ( (28 - 20)//2, (28 - 20)//2 )
new_image.paste(image, upper_left)

transform = transforms.ToTensor()
image = transform(new_image).unsqueeze(0).to(device)

plt.imshow(image.squeeze(0).squeeze(0).cpu(), cmap='gray')
plt.show()

with torch.no_grad():
    output = model(image)
    prediction = output.argmax(1).item()

print(f"Predicted digit: {prediction}")