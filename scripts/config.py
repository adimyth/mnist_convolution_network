from attrdict import AttrDict

configs = {
    "RANDOM_STATE": 42,
    "EPOCHS": 10,
    "BATCH_SIZE": 64,
    "NUM_CLASSES": 10,
    "TRAIN_PATH": "../data/train.csv",
    "TEST_PATH": "../data/test.csv",
    "SUBMISSION_PATH": "../data/sample_submission.csv",
}
configs = AttrDict(configs)
