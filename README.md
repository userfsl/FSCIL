## Datasets and pretrained models
Please follow the guidelines in [LIMIT](https://github.com/zhoudw-zdw/TPAMI-Limit) to prepare the dataset.

We also provide the pre-trained models in (https://drive.google.com/file/d/1EqQ8C8vNmZ3ez_qiSGnkp2zZba9Q45z0/view?usp=sharing). Please place it under `checkpoint/` folder.

## Training scripts
If you want to use the models pre-trained on the base session, you can run:

    $  python train.py -project fscil -dataset cub200  -episode_way 100 -episode_shot 10 -episode_query 15 -low_way 40 -low_shot 5 -lr_base 0.005 -lrg 0.05 -step 10 -gamma 0.5 -lr_new 0.05 -decay 0.0005 -gpu 1,2,3 -model_dir checkpoint/session0_max_acc.pth

(Optional) If you want to directly use the backbone pre-trained on the ImageNet, you can run:

    $ python train.py -project fscil -dataset cub200  -episode_way 100 -episode_shot 10 -episode_query 15 -low_way 40 -low_shot 5 -lr_base 0.005 -lrg 0.05 -step 10 -gamma 0.5 -lr_new 0.05 -decay 0.0005 -gpu 1,2,3


Note that different devices may get slightly different results. More detailed information will be released upon acceptance.
