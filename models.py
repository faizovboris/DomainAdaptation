import torch
import torch.nn as nn

import dann_config
from blocks import GradientReversalLayer


def parse_layers(model, layer_ids):
    """
    Args:
        model (nn.Module) - nn.Sequential model
        layer_ids (list of int) - required model outputs
    Return:
        layers (list of nn.Module)- model splitted in parts

    Function for splitting nn.Sequential model in separate
    layers, which indexed by numbers from the list layer_ids.
    It will be required to obtain activations of intermediate model layers
    and calculate loss function between intermediate layers
    """
    separate_layers = list(model)
    layers = []
    for i in range(len(layer_ids)):
        if i == 0:
            start_layer = 0
        else:
            start_layer = layer_ids[i - 1] + 1
        layers.append(nn.Sequential(*separate_layers[start_layer:layer_ids[i] + 1]))
    if len(separate_layers) > layer_ids[-1] + 1:
        layers.append(nn.Sequential(*separate_layers[layer_ids[-1] + 1:]))        
    return layers


def get_base_model():
    """
    Return:
        features (nn.Module) - convolutional model part for feature extracting
        pooling (nn.Module) - model pooling layers
        classifier (nn.Module) - model fully connected layers
        pooling_ftrs (int) - number of activations at the output of "pooling"
        pooling_output_side (int) - side of feature map at the output of "pooling"
        layers (list of nn.Module) - model splitted in parts

    Returns three parts of model — the convolutional part to extract features,
    part with pooling and part with fully connected model layers.
    Can return these parts with pre-trained weights for standard architecture.
    """

    if dann_config.model_backbone == "alexnet":
        from torchvision.models import alexnet
        model = alexnet(pretrained=dann_config.backbone_pretrained)
        features, pooling, classifier = model.features, model.avgpool, model.classifier
        classifier[-1] = nn.Linear(4096, dann_config.classes_cnt)
        classifier_layer_ids = [1, 4, 6]
        pooling_ftrs = 256
        pooling_output_side = 6
    elif dann_config.model_backbone == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=dann_config.backbone_pretrained)
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        pooling = model.avgpool
        classifier = nn.Sequential(nn.Linear(2048, dann_config.classes_cnt))
        classifier_layer_ids = [0]
        pooling_ftrs = 2048
        pooling_output_side = 1
    elif dann_config.model_backbone == 'vanilla_dann' and dann_config.backbone_pretrained == False:
        hidden_size = 64
        pooling_output_side = (dann_config.image_side - 12) // 4

        features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=5),
            nn.BatchNorm2d(hidden_size),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=5),
            nn.BatchNorm2d(hidden_size),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        pooling = nn.Sequential()
        classifier = nn.Sequential(
            nn.Linear(hidden_size * pooling_output_side * pooling_output_side, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, dann_config.classes_cnt),
        )
        classifier_layer_ids = [0, 4, 7]
        pooling_ftrs = hidden_size
    else:
        raise RuntimeError("model %s with pretrained = %s, does not exist" % (dann_config.model_backbone, dann_config.backbone_pretrained))

    if dann_config.loss_need_intermediate_layers:
        classifier = nn.ModuleList(parse_layers(classifier, classifier_layer_ids))
    else:
        classifier = nn.ModuleList([classifier])
    return features, pooling, classifier, pooling_ftrs, pooling_output_side


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass


class DANNModel(BaseModel):
    def __init__(self):
        super(DANNModel, self).__init__()
        self.features, self.pooling, self.class_classifier, pooling_ftrs, pooling_output_side = get_base_model()
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(pooling_ftrs * pooling_output_side * pooling_output_side, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (map of tensors) - map with model output tensors
        """
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)
        
        output_classifier = features
        classifier_layers_outputs = []
        for block in self.class_classifier:
            output_classifier = block(output_classifier)
            classifier_layers_outputs.append(output_classifier)
        
        reversed_features = GradientReversalLayer.apply(features, dann_config.gradient_reversal_layer_alpha)
        output_domain = self.domain_classifier(reversed_features)
        
        output = {
            "class": output_classifier,
            "domain": output_domain,
        }
        if dann_config.loss_need_intermediate_layers:
            output["classifier_layers"] = classifier_layers_outputs

        return output
    
    def predict(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        return self.forward(input_data)["class"]
