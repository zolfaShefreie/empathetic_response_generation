"""source is https://github.com/declare-lab/exemplary-empathy/blob/main/dpr_exempler_retriever.py"""
import time
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pandas as pd
import os
import torch
import numpy as np

from settings import DPR_ENCODER_PATH


class ExampleRetriever:

    BATCH_SIZE = 32
    EXAMPLE_KEY_NAME = 'examples'

    def __init__(self, train_df: pd.DataFrame, ctx_key_name, qs_key_name, conv_key_name):

        self.ctx_key_name = ctx_key_name
        self.qs_key_name = qs_key_name
        self.conv_key_name = conv_key_name

        # models initials
        # from_pretrained gets ssl exception so i put the code inside the try catch
        count = 10
        while True:
            try:
                self.CTX_MODEL = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
                self.QS_MODEL = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
                self.CTX_TOKENIZER = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
                self.QS_TOKENIZER = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
                break
            except Exception as e:
                print(e)
                count -= 1
                if count == 0:
                    raise e
                time.sleep(1)
        self.QS_TOKENIZER.truncation_side = "left"

        self.load_fine_tuned_dpr()
        self.CTX_MODEL.eval()
        self.QS_MODEL.eval()

        # train dataset initials
        self.train_df = train_df
        self.train_ctx = self.embeddings_for_batch_sentence(sentences=list(train_df[self.ctx_key_name]), is_ctx=True)

        # faiss initials
        dim = self.train_ctx.shape[1]
        self.index_flat = faiss.IndexFlatIP(dim)
        try:
            faiss.StandardGpuResources()
            self.CAN_USE_GPU_VERSION = True and torch.cuda.is_available()
        except Exception:
            self.CAN_USE_GPU_VERSION = False

        if self.CAN_USE_GPU_VERSION:
            res = faiss.StandardGpuResources()
            self.gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index_flat)
            self.gpu_index_flat.add(self.train_ctx)
        else:
            self.index_flat.add(self.train_ctx)

    def compute_examples(self, query: dict, query_embedding: np.array):
        """
        compute examples for one record
        :param query:
        :param query_embedding:
        :return:
        """
        k = 2047

        D, I = self.index_flat.search(x=query_embedding, k=k) if not self.CAN_USE_GPU_VERSION \
            else self.gpu_index_flat.search(x=query_embedding, k=k)

        examples = list(self.train_df[self.ctx_key_name])
        record_examples = []
        emotions, conv_id = query["xReact"], query[self.conv_key_name]
        candidate_indices = set(self.train_df[self.train_df.apply(lambda x: len(set(x['xReact']).intersection(set(emotions))) != 0 and x[self.conv_key_name] != conv_id, axis=1)].index)

        retrieved, matches = np.array(I[0]), []
        for item in retrieved:
            if item in candidate_indices:
                matches.append(item)
            if len(matches) == 10:
                break

        record_examples += [examples[ind] for ind in matches]
        query[self.EXAMPLE_KEY_NAME] = record_examples
        return query

    def embeddings_for_batch_sentence(self, sentences: list, is_ctx: bool = True) -> np.array:
        """
        get embeddings for
        :param sentences:
        :param is_ctx:
        :return:
        """
        tokenizer = self.CTX_TOKENIZER if is_ctx else self.QS_TOKENIZER
        model = self.CTX_MODEL if is_ctx else self.QS_MODEL
        embeddings = []

        for j in range(0, len(sentences), self.BATCH_SIZE):
            batch = tokenizer(sentences[j:j + self.BATCH_SIZE], truncation=True, padding=True, max_length=512, return_tensors="pt")
            input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]
            attention_mask = batch["attention_mask"].cuda() if torch.cuda.is_available() else batch["attention_mask"]
            with torch.no_grad():
                output = model(input_ids, attention_mask)
            embeddings.append(output.pooler_output)
        return np.array(torch.cat(embeddings)) if not torch.cuda.is_available() else np.array(torch.cat(embeddings).cpu())

    def embeddings_for_one_sentence(self, sentence: str, is_ctx: bool = True) -> np.array:
        """
        get embedding of one sentence
        :param sentence:
        :param is_ctx:
        :return:
        """
        tokenizer = self.CTX_TOKENIZER if is_ctx else self.QS_TOKENIZER
        model = self.CTX_MODEL if is_ctx else self.QS_MODEL
        model.eval()
        tokenized_sentence = tokenizer(sentence, truncation=True, padding=True, max_length=512, return_tensors="pt")
        if torch.cuda.is_available():
            tokenized_sentence = {k: v.cuda() for k, v in tokenized_sentence.items()}
        with torch.no_grad():
            output = model(tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']).pooler_output
            return np.array(output) if not torch.cuda.is_available() else np.array(output.cpu())

    def load_fine_tuned_dpr(self):
        """
        load fine_tuned_dpr for CTX_MODEL and QS_MODEL
        :return:
        """
        if DPR_ENCODER_PATH and os.path.exists(DPR_ENCODER_PATH):
            weights = torch.load(DPR_ENCODER_PATH)["model_dict"]
            ctx_model_state_dict = self.CTX_MODEL.state_dict()
            qs_model_state_dict = self.QS_MODEL.state_dict()

            for key in ctx_model_state_dict:
                if "weight" in key or "bias" in key:
                    new_key = key.replace("ctx_encoder.bert_model", "ctx_model")
                    ctx_model_state_dict[key] = weights[new_key]

            for key in qs_model_state_dict:
                if "weight" in key or "bias" in key:
                    new_key = key.replace("question_encoder.bert_model", "question_model")
                    qs_model_state_dict[key] = weights[new_key]

            self.CTX_MODEL.load_state_dict(ctx_model_state_dict)
            self.QS_MODEL.load_state_dict(qs_model_state_dict)

    def __call__(self, data: dict):
        """
        run process function
        :param data:
        :return:
        """
        qs_embedding = self.embeddings_for_one_sentence(data[self.qs_key_name], is_ctx=False)
        return self.compute_examples(query=data, query_embedding=qs_embedding)
