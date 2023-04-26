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


def get_dataloaders(data_dir='flowers', batch_size=64):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_std)
        ]),
    }
    data_transforms['test'] = data_transforms['valid']

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(test_dir,  transform=data_transforms['test']),
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False),
    }
    return image_datasets, dataloaders


def load_arch(model_name, hidden_layers):
    #weigths = getattr(models, model_name.upper()+'_Weights').DEFAULT
    #model = getattr(models, model_name)(weights=weigths)
    model = getattr(models, model_name)(pretrained=True)
    if isinstance(model.classifier, list):
        n_features = model.classifier[0].in_features
    else:
        n_features = model.classifier.in_features
    model_classifier = Network(n_features, 102, hidden_layers)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = model_classifier
    return model


def train(model, dataloaders, epochs, learning_rate, optimizer=None):
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    if optimizer is None:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer


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
