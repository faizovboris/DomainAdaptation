import torch
import torch.nn as nn

import dann_config
import backbone_models
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


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass


class DANNModel(BaseModel):
    def __init__(self):
        super(DANNModel, self).__init__()
        self.features, self.pooling, self.class_classifier, \
            pooling_ftrs, pooling_output_side = backbone_models.get_backbone_model()
        
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
