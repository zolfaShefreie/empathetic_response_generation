import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)

DATASET_CACHE_PATH = "./data/cache"

MELD_DATASET_PATH = env('MELD_DATASET_PATH', default="./")
EMPATHY_CLASSIFIER_MODELS_PATH = env('EMPATHY_CLASSIFIER_MODELS_PATH', default="./")

#train settings
DEFAULT_SAVE_DIR_PREFIX = "./models_checkpoint"

# huggingface hub settings
# text generator using empatheticdialogues
HUB_TEXT_MODEL_ID = env('HUB_TEXT_MODEL_ID', default=None),
HUB_TEXT_PRIVATE_REPO = env('HUB_TEXT_PRIVATE_REPO', default=False)

# multimodal emotion classifier using meld dataset
HUB_CLASSIFIER_MODEL_ID = env('HUB_CLASSIFIER_MODEL_ID', default=None),
HUB_CLASSIFIER_PRIVATE_REPO = env('HUB_CLASSIFIER_PRIVATE_REPO', default=False)

HUB_ACCESS_TOKEN = env('HUB_ACCESS_TOKEN', default=None)


DPR_ENCODER_PATH = None
