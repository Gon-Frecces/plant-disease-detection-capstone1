import torch
model = torch.load("models/plant_disease_classifier.pth", map_location="cpu")
print(type(model))
