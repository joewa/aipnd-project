import os
import json
import argparse
import utils


argparser = argparse.ArgumentParser(
        description='Predict a flower name from an image.',
        usage='python predict.py ./flowers/valid/1/img.jpg'
    )
argparser.add_argument('image_path', type=str, help='path to image file')
argparser.add_argument('checkpoint', type=str, help='path to Checkpoint file')
argparser.add_argument('--gpu', action='store_true', default=False, help='use GPU for prediction if available')
argparser.add_argument('--topk', type=int, default=1, help='return top K most likely classes')
argparser.add_argument('--category_names', type=str, default='cat_to_name.json',
                    help='mapping file of class id to real names. default "cat_to_name.json"',)
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

    # load in a mapping from category label to category name
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    my_checkpoint = utils.load_checkpoint(args.checkpoint)
    model = my_checkpoint['model']

    top_p, top_class = utils.predict(args.image_path, model, topk=args.topk)
    top_p_list = top_p.squeeze().tolist()
    top_class_names = [cat_to_name[c] for c in top_class]
    if not isinstance(top_p_list, list):
        top_p_list = [top_p_list]

    for n in range(args.topk):
        print('Prediction:{0}; Probability:{1}'.format(top_class_names[n], top_p_list[n]))
    pass


if __name__ == '__main__':
    main(args)
