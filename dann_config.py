weight_domain_loss = 1
weight_prediction_loss = 1 # todo: find actual values
unk_value = -100 # torch default
target_domain_idx = 1 

loss_need_intermediate_layers = False
classes_cnt = 31
model_type = "DANN"
model_backbone = "alexnet" # alexnet resnet50 vanilla_dann
backbone_pretrained = True
gradient_reversal_layer_alpha = 1.0
image_side = 256
