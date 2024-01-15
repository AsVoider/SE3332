import argparse

import torch

import lab3_data_scratch
from train_eva import transform
from main import num_frames, num_classes, best_weights, last_weights, acc_weight

import my_model, pretrained_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="select model")
    parser.add_argument('--model_select', type=str, default="my_model",
                        help='model select from my_model, pre_trained')
    parser.add_argument('--pt_select', type=str, default="last",
                        help='pt last ? best ? acc')
    parser.add_argument('--video_path', type=str, default=None,
                        help="select path")

    args = parser.parse_args()
    assert not (args.video_path is None)
    data = lab3_data_scratch.read_radio(args.video_path, transform, num_frames)
    data = data.unsqueeze(0)
    data = data.to('cuda')
    model = my_model.MyModel(num_classes=num_classes, hidden_size=128, num_lstm=2) if args.model_select == "my_model" \
        else pretrained_model.ClassificationModel(num_classes=num_classes, hidden_size=128, num_lstm_layers=2)
    model.to('cuda')
    load_path = last_weights
    if 'best' == args.pt_select:
        load_path = best_weights
    else:
        load_path = acc_weight
    model.load_state_dict(torch.load(load_path))
    model.eval()
    output = model(data)
    output = output.reshape(-1)
    _, idx = torch.max(output, dim=0)
    print(f'predict class is {idx.item()}')
    print(output)
