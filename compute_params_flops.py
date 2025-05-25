from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.model_utils.build_model import build_model
from fgir_kd.data_utils.build_dataloaders import build_dataloaders
from fgir_kd.train_utils.misc_utils import count_params_single, count_params_trainable, count_flops


def main():
    args = parse_inference_args()
    args.try_fused_attn = False

    _, _, test_loader = build_dataloaders(args)

    model = build_model(args)

    flops = count_flops(model, args.image_size, args.device)
    no_params = round((count_params_single(model) / 1e6), 4)
    no_params_trainable = round((count_params_trainable(model) / 1e6), 4)

    print('dataset_name,model_name,flops,no_params,no_params_trainable')
    line = f'{args.dataset_name},{args.model_name},{flops},{no_params},{no_params_trainable}'
    print(line)
    return 0


if __name__ == "__main__":
    main()
