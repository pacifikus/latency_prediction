import pandas as pd
import torch


def get_conv_kernel_size(model, kernel_size):
    convs = [layer.kernel_size[0] == kernel_size for name, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]
    return len(convs)


def get_linear_input_features(model):
    linear_layers = [layer.in_features for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    return linear_layers[0]


def get_linear_output_features(model):
    linear_layers = [layer.out_features for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    return linear_layers[-1]


def get_layer_type_count(model, layer_type):
    res = 0
    for name, layer in model.named_modules():
        if isinstance(layer, layer_type):
            res += 1
    return res


def get_features(config):
    data = pd.read_csv(config['data']['model_result_df_path'])
    result = []
    for _, row in data.iterrows():
        try:
            model_file_name = row['model_uuid']
            model = torch.load(
                f"{config['data']['nn_models_path']}{model_file_name}"
            )
            res = {
                'model_uuid': model_file_name,
                'conv_cnt': get_layer_type_count(model, torch.nn.Conv2d),
                'linear_cnt': get_layer_type_count(model, torch.nn.Linear),
                'relu_cnt': get_layer_type_count(model, torch.nn.ReLU),
                'lrelu_cnt': get_layer_type_count(model, torch.nn.LeakyReLU),
                'avg_pooling_cnt': get_layer_type_count(model, torch.nn.AvgPool2d),
                'max_pooling_cnt': get_layer_type_count(model, torch.nn.MaxPool2d),
                'out_features': get_linear_output_features(model),
                'linear_input_features': get_linear_input_features(model),
                'conv_kernel_size_2': get_conv_kernel_size(model, 2),
                'conv_kernel_size_3': get_conv_kernel_size(model, 3),
                'time': row['time']
            }
            result.append(res)
        except:
            print('error')
    pd.DataFrame(result).to_csv(
        config['data']['processed_df_path'],
        index=False)
