from .resnet import ResNet9
from .vgg import VGG
from .simplecnn import SimpleCNN
from .wrn import wrn_16_2
from .resnet_fmnist import resnet8_fmnist
import torchvision.models as models
import torch.nn as nn


def get_model(data, args):
    if data == 'cifar10':
        model = wrn_16_2(num_classes=10)
    elif data == 'cifar100':
        model = VGG('VGG9',num_classes=100)
        # model = wrn_16_2(num_classes=100)
    elif data == 'tinyimagenet':
        # model = get_resnet18_64x64()
        model = get_resnet50_64x64()
    elif data == 'fmnist':
        # model = SimpleCNN(output_dim=256, n_classes=10, width=2)
        model = resnet8_fmnist()
    else:
        raise NotImplementedError(f'Model for dataset {data} is not implemented.')

    return model

def get_resnet18_64x64(num_classes=200):
    """
    Returns a ResNet-18 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-18 model.
    """
    # Load the ResNet-18 model pretrained on ImageNet
    model = models.resnet18(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet50_64x64(num_classes=200):
    """
    Returns a ResNet-50 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-50 model.
    """
    # Load the ResNet-50 model pretrained on ImageNet
    model = models.resnet50(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet101_64x64(num_classes=200):
    """
    Returns a ResNet-101 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-50 model.
    """
    # Load the ResNet-50 model pretrained on ImageNet
    model = models.resnet101(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model