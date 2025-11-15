import os
import torch


class CFG:

    # preprocess
    DROP_HEAD_SECONDS = 6.7

    # outlier removal
    OUTLIER_METHOD = 'IQR' # options: 'none', 'IQR'
    NORMALIZE_METHOD = 'minmax' # 'none', 'zscore', 'minmax', 'robust'
    LOWPASS_FILTER = 'butterworth' # 'none', 'butterworth', 'moving_average'
    FEATURE_SELECTION = 'none' # 'none', 'pca', 'robustpca', 'kernelpca'

    # split seed
    RANDOM_SEED = 42 

    # IQR window for outlier removal
    OUTLIER_WINDOW = 11

    # lowpass filter
    CUTOFF = 300
    FS = 10000
    BUTTER_ORDER = 2
    MA_WINDOW = 7

    # feature selection
    PCA_COMPONENTS = 8

    # sliding window
    WINDOW_LEN = 200
    STRIDE = 100

    # dataset split
    SPLIT_RATIO = (0.8, 0.1, 0.1)

    COLUMNS = ["Time","Ipv","Vpv","Vdc","ia","ib","ic","va","vb","vc","Iabc","If","Vabc","Vf","label"]



    # preprocess_methods

    # kernel pca
    NB_SAMPLES_KPCA = 3000
    KERNEL_KPCA = 'rbf'
    GAMMA_KPCA = 0.1



