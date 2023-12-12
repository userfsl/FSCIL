## Datasets and pretrained models
Please follow the guidelines in [LIMIT](https://github.com/zhoudw-zdw/TPAMI-Limit) to prepare the dataset.
We also provide the pretrained models under `checkpoint/` folder.

## Training scripts
If you want to use the models pre-trained on the base session, you can run:
    $ python train.py -project fscil -dataset cub200  -episode_way 100 -episode_shot 10 -episode_query 15 -low_way 40 -low_shot 5 -lr_base 0.005 -lrg 0.05 -step 10 -gamma 0.5 -lr_new 0.05 -decay 0.0005 -gpu 0,1,2,3,4,5,6,7 -model_dir checkpoint/session0_max_acc.pth
If you want to directly use the backbone pre-trained on the ImageNet, you can run:
    $ python train.py -project fscil -dataset cub200  -episode_way 100 -episode_shot 10 -episode_query 15 -low_way 40 -low_shot 5 -lr_base 0.005 -lrg 0.05 -step 10 -gamma 0.5 -lr_new 0.05 -decay 0.0005 -gpu 0,1,2,3,4,5,6,7

More detailed information will be released upon acceptance.
