import torch
import torch.nn as nn

from blocks import GradientReversalLayer

def parse_layers(layers, layer_ids):
    """
    Фунцкия для для разделения одной nn.Sequential-части модели
    на составные части между раздельными слоями с номерами из списка layer_ids.
    Понадобится, чтобы получать активации в промежуточных слоях модели при
    наличии функции потерь между ними.
    """
    separate_layers = list(layers)
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


def get_base_model(config):
    """
    Возвращает три составные части модели - сверточную часть для извлечения признаков,
    часть для пулинга и часть для полносвязных слоёв модели. Может вернуть эти части как с
    предобученными весами для стандартных архитектур, так и для произволных
    собственных архитектур.
    Дополнительно возвращает features_cnt, features_output_side - число активаций
    и размер карты на выходе из части для извлечения признаков.
    """
    if config["model"]["base"] == "alexnet":
        from torchvision.models import alexnet
        model = alexnet(pretrained=config["model"]["pretrained"])
        features, pooling, classifier = model.features, model.avgpool, model.classifier
        classifier[-1] = nn.Linear(4096, config["classes_cnt"])
        classifier_layer_ids = [1, 4, 6]
        features_cnt = 256
        features_output_side = 6
    elif config["model"]["base"] == 'vanilla_dann' and config["model"]["pretrained"] == False:
        hidden_size = 64
        features_output_side = (config["image_side"] - 12) // 4

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
            nn.Linear(hidden_size * features_output_side * features_output_side, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, config["classes_cnt"]),
        )
        classifier_layer_ids = [0, 4, 7]
        features_cnt = hidden_size
    else:
        raise RuntimeError("model %s with pretrained = %s, does not exist" % (config["model"]["base"], config["model"]["pretrained"]))

    if config["loss"]["need_classifier_layers"]:
        classifier = nn.ModuleList(parse_layers(classifier, classifier_layer_ids))
    else:
        classifier = nn.ModuleList([classifier])
    return features, pooling, classifier, features_cnt, features_output_side


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass


class DANNModel(BaseModel):
    def __init__(self, config):
        super(DANNModel, self).__init__()
        self.config = config

        self.features, self.pooling, self.class_classifier, features_cnt, features_output_side = get_base_model(config)
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(features_cnt * features_output_side * features_output_side, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input_data):
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)
        
        output_classifier = features
        classifier_layers_outputs = []
        for block in self.class_classifier:
            output_classifier = block(output_classifier)
            classifier_layers_outputs.append(output_classifier)
        
        reversed_features = GradientReversalLayer.apply(features, self.config["model"]["reversal_layer_alpha"])
        output_domain = self.domain_classifier(reversed_features)
        
        output = {
            "class": output_classifier,
            "domain": output_domain,
        }
        if self.config["loss"]["need_classifier_layers"]:
            output["classifier_layers"] = classifier_layers_outputs

        return output
    
    def predict(self, input_data):
        """
        Функция, которая используется уже в процессе тестирования при
        целевом использовании модели, а не в процессе обучения.
        """
        return self.forward(input_data)["class"]

