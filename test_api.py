from PIL import Image
import numpy as np
import requests
import json

# Create a test image (random noise = synthetic)
img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
img.save('test_img.jpg', quality=90)

# Upload and analyze
with open('test_img.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/api/analyze', files={'image': ('test.jpg', f, 'image/jpeg')})

data = r.json()
print("Success:", data["success"])
print("Verdict:", data["verdict"]["verdict"])
print("Confidence:", round(data["verdict"]["confidence"], 1), "%")
print("Score:", round(data["verdict"]["score"], 3))
print()
for key, val in data["analysis"].items():
    name = val["name"]
    score = round(val["score"]*100)
    print(f"  {name}: {score}%")
