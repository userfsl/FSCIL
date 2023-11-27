## Datasets and pretrained models
Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare the dataset.
We also provide the pretrained models under `checkpoint/` folder.

## Training scripts

    $ python train.py -project fscil -dataset cub200  -episode_way 100 -episode_shot 10 -episode_query 15 -low_way 60 -low_shot 5 -lr_base 0.005 -lrg 0.05 -step 10 -gamma 0.5 -lr_new 0.05 -decay 0.0005 -gpu 0,1,2,3,4,5,6,7

The logs and results are placed in `log/` folder.

More detailed information will be released upon acceptance.
