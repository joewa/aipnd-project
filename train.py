import os
import json
import argparse
import utils
import torch


argparser = argparse.ArgumentParser(
        description='Train a NN to predict flower names from an images.',
        usage='python train.py data_directory'
    )
argparser.add_argument('data_directory', type=str, help='must contain subdirectoreis train, valid and test')
argparser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='path to Checkpoint file')
argparser.add_argument('--gpu', action='store_true', default=False, help='use GPU for prediction if available')
argparser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='name of the checkpoint file')
argparser.add_argument('--arch', type=str, default='densenet121',
                    help='NN architecture from https://pytorch.org/vision/stable/models.html',)
argparser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate. default=0.01')
argparser.add_argument('--epochs', type=int, default=3, help='number of epochs')
argparser.add_argument('--hidden_units_list', nargs='+', type=int, required=True, help='list of hidden units per layer')
args = argparser.parse_args()


def main(args):
    use_gpu = args.gpu
    if use_gpu:
        if utils.device.type != 'cpu':
            print('Info -- Using GPU: {}'.format(utils.device.type))
        else:
            print('Info -- Cannot use GPU, falling back to CPU')
    else:
        print('Info -- Using CPU')
        utils.device = utils.torch.device('cpu')

    image_datasets, dataloaders = utils.get_dataloaders(data_dir=args.data_directory, batch_size=64)
    # Load the model's architecture
    model = utils.load_arch(args.arch, args.hidden_units_list)

    model, optimizer = utils.train(model, dataloaders, args.epochs, args.learning_rate)

    model.class_to_idx = image_datasets['train'].class_to_idx

    # Save the full model
    print('Info -- Saving checkpoint {}'.format(args.checkpoint))
    checkpoint = {
        #'input_size': 784,
        #'output_size': 10,
        'model_module': type(model).__module__,
        'model_name': args.arch,  # type(model).__name__,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': {
            'state_dict': model.classifier.state_dict(),
            'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
            'input_size': model.classifier.hidden_layers[0].in_features,
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': running_loss,
        },
    }
    torch.save(checkpoint, args.checkpoint)

    pass


if __name__ == '__main__':
    main(args)
