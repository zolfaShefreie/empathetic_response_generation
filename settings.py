import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)

DATASET_CACHE_PATH = "./data/cache"

MELD_DATASET_PATH = env('MELD_DATASET_PATH', default="./")
BMEDIALOGUES_PATH = env("BMEDIALOGUES_PATH", default=None)
EMPATHY_CLASSIFIER_MODELS_PATH = env('EMPATHY_CLASSIFIER_MODELS_PATH', default="./")

#train settings
DEFAULT_SAVE_DIR_PREFIX = "./models_checkpoint"

# huggingface hub settings
# text generator using empatheticdialogues
HUB_EMO_TEXT_MODEL_ID = env('HUB_TEXT_MODEL_ID', default=None),
HUB_EMO_TEXT_PRIVATE_REPO = env('HUB_TEXT_PRIVATE_REPO', default=False)

# multimodal emotion classifier using meld dataset
HUB_CLASSIFIER_MODEL_ID = env('HUB_CLASSIFIER_MODEL_ID', default=None),
HUB_CLASSIFIER_PRIVATE_REPO = env('HUB_CLASSIFIER_PRIVATE_REPO', default=False)

# text generator using text and audio modals
HUB_BIMODEL_ID = env("HUB_MODEL_ID", default=None)
HUB_BIMODEL_PRIVATE_REPO = env("HUB_MODEL_PRIVATE_REPO", default=None)

# text generator using text of BiMEmpDialogues dataset
HUB_TEXT_MODEL_ID = env("HUB_TEXT_MODEL_ID", default=None)
HUB_TEXT_PRIVATE_REPO = env("HUB_TEXT_PRIVATE_REPO", default=None)

HUB_ACCESS_TOKEN = env('HUB_ACCESS_TOKEN', default=None)


DPR_ENCODER_PATH = None
