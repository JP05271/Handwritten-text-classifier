import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
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

# Invert colors (white digit on black background)
image = ImageOps.invert(image)

# Threshold to pure black/white (optional but helps for bounding box)
image = image.point(lambda x: 0 if x < 128 else 255, '1')

# Find bounding box of the digit
bbox = image.getbbox()
if bbox:
    image = image.crop(bbox)

# Resize to 20x20
image = image.resize((20, 20), Image.Resampling.LANCZOS)

# Create a new blank 28x28 black image and paste the 20x20 digit into center
new_image = Image.new('L', (28, 28), 0)
upper_left = ( (28 - 20)//2, (28 - 20)//2 )
new_image.paste(image, upper_left)

# To tensor
transform = transforms.ToTensor()
image = transform(new_image).unsqueeze(0).to(device)

# Optional: Visualize the processed image
plt.imshow(image.squeeze(0).squeeze(0).cpu(), cmap='gray')
plt.show()

# Predict
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(1).item()

print(f"Predicted digit: {prediction}")