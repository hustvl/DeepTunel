import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=[255.0,255.0,255.0],
        # WEIGHTS="/home/lianjunwu/code/R-50.pkl", #offline
        RESNETS=dict(DEPTH=50)
    ),
    DATASETS=dict(
        TRAIN=('DIAOWANG_2022_new_Cross_Individual_train',),
        TEST=('DIAOWANG_final_cross_anns',),
    ),
    SOLVER=dict(
            CHECKPOINT_PERIOD=28604 * 2,
            LR_SCHEDULER=dict(
                MAX_ITER=28604 * 12, #20828 images for 12 epoch
                STEPS=(28604 * 8, 28604 * 11),
            ),
            OPTIMIZER=dict(
                BASE_LR=0.01, # learning rate in original config is used for 8 GPUs 16 total batch;
            ),
            IMS_PER_DEVICE=2,
            
    ),
    TEST=dict(
        EVAL_PERIOD=20,
        EVALUATION_TYPE="MidogEvaluator",
    ),
    TRAINER=dict(
        NAME="DiaoWangRunner",
    ),
    DATALOADER=dict(
        # Number of data loading threads
        NUM_WORKERS=8,
        # ASPECT_RATIO_GROUPING=True,
        # # Default sampler for dataloader
        # SAMPLER_TRAIN="ClassAwareSampler",
        # # Use infinite wraper for sampler or not.
        # ENABLE_INF_SAMPLER=True,
        # # Repeat threshold for RepeatFactorTrainingSampler
        # REPEAT_THRESHOLD=0.0,
        # # If True, the dataloader will filter out images that have no associated
        # # annotations at train time.
        # FILTER_EMPTY_ANNOTATIONS=False,
        # NUM_SAMPLE_CLASS=[2,2]
        ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(512,), max_size=512, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=512, max_size=512, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
