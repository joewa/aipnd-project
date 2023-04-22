import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json


# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


norm_means = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]#
img_size = 224


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            # x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def load_checkpoint(filepath):
    import importlib

    checkpoint = torch.load(filepath)

    # Load the pre-trained model
    model_module = importlib.import_module(checkpoint['model_module'])
    model = getattr(model_module, checkpoint['model_name'])()

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


    # TODO: load the pre-trained model
    model_classifier = Network(
        checkpoint['classifier']['input_size'],
        102,  # checkpoint['output_size'],
        checkpoint['classifier']['hidden_layers'])
    model.classifier = model_classifier

    model.load_state_dict(checkpoint['state_dict'])
    #model_classifier.load_state_dict(checkpoint['classifier']['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Optimizer state if we want to continue the training
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    #optimizer.load_state_dict(checkpoint['classifier']['optimizer_state_dict'])

    return {'model': model, 'optimizer': optimizer}


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_std)
        ])
    pil_img = Image.open(image)
    pt_img = img_transform(pil_img).float()

    return pt_img


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    pil_img = process_image(image_path)
    with torch.no_grad():
        input_img = pil_img.to(device)
        input_img.unsqueeze_(0)
        model.to(device)
        logps = model.forward(input_img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    idx_to_class = {model.class_to_idx[c]: c for c in model.class_to_idx.keys()}
    return top_p.cpu().numpy(), np.array([idx_to_class[idx] for idx in top_class.cpu().numpy()[0]])
