import environ
import os

env = environ.Env()
env_path = "./.env"
environ.Env.read_env(env_path)

DATASET_CACHE_PATH = "./data/cache"


#train settings
DEFAULT_SAVE_DIR_PREFIX = "./models_checkpoint"

# huggingface hub settings
HUB_MODEL_ID = {
    'roberta_shared': env('HUB_MODEL_ID_SHARED', default=None),
    'roberta_gpt2': env('HUB_MODEL_ID_GPT', default=None)
}
HUB_PRIVATE_REPO = {
    'roberta_shared': env('HUB_PRIVATE_REPO_SHARED', default=False),
    'roberta_gpt2': env('HUB_PRIVATE_REPO_GPT', default=None)
}

HUB_ACCESS_TOKEN = env('HUB_ACCESS_TOKEN', default=None)


DPR_ENCODER_PATH = None