from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv import Config

import os
import torch


mmcls_path = os.environ['MMCLS_PATH']

config_file = os.path.join(
    mmcls_path, 'configs/resnet/resnet50_b32x8_imagenet.py'
)
model_file = os.path.join(
    mmcls_path, 'models/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
)
device = 'cuda:0'

img_file = os.path.join(mmcls_path, 'demo/demo.JPEG')


def predict():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='image file')
    # parser.add_argument('config', help='config file')
    # parser.add_argument('checkpint', help='checkpoint file')
    # parser.add_argument('--device', default='cuda:0', help='device used for inference')

    # args = parser.parse_args()

    print(config_file)
    print(model_file)
    print(img_file)

    model = init_model(config_file, model_file, device=device)
    result = inference_model(model, img_file)
    show_result_pyplot(model, img_file, result)

    # Pack image info into a dict
    data = dict(img_info=dict(filename=img_file), img_prefix=None)
    # Parse the test pipeline
    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)
    # Process the image
    data = test_pipeline(data)

    # Scatter to specified GPU
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    # Forward the model
    with torch.no_grad():
        features = model.extract_feat(data['img'])

    # Show the feature, it is a 1280-dim vector
    print(features.shape)


def finetune():
    cfg = Config.fromfile('configs/resnet/resnet50_b32x8_imagenet.py')
    cfg.model.head.num_classes = 2
    cfg.model.head.topk = 1

    # Modify the number of workers according to your computer
    cfg.data.samples_per_gpu = 32
    cfg.data.workers_per_gpu = 2
    # Modify the image normalization configs
    cfg.img_norm_cfg = dict(
        mean=[124.508, 116.050, 106.438],
        std=[58.577, 57.310, 57.437],
        to_rgb=True,
    )
    # Specify the path to training set
    cfg.data.train.data_prefix = (
        'data/cats_dogs_dataset/training_set/training_set'
    )
    cfg.data.train.classes = 'data/cats_dogs_dataset/classes.txt'
    # Specify the path to validation set
    cfg.data.val.data_prefix = 'data/cats_dogs_dataset/val_set/val_set'
    cfg.data.val.ann_file = 'data/cats_dogs_dataset/val.txt'
    cfg.data.val.classes = 'data/cats_dogs_dataset/classes.txt'
    # Specify the path to test set
    cfg.data.test.data_prefix = 'data/cats_dogs_dataset/test_set/test_set'
    cfg.data.test.ann_file = 'data/cats_dogs_dataset/test.txt'
    cfg.data.test.classes = 'data/cats_dogs_dataset/classes.txt'
    # Modify the metric method
    cfg.evaluation['metric_options'] = {'topk': (1)}


if __name__ == '__main__':
    predict()
