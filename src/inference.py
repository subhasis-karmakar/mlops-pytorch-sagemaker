import torch
from model import SimpleCNN

def model_fn(model_dir):
    model = SimpleCNN()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
    model.eval()
    return model

def predict_fn(input_data, model):
    with torch.no_grad():
        return model(input_data)