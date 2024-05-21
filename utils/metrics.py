from transformers import EvalPrediction
from collections import Counter
import evaluate
import numpy as np


class Metrics:
    PPL = evaluate.load("perplexity", module_type="metric")
    BLEURT = evaluate.load("bleurt", module_type="metric")
    ROUGE = evaluate.load('rouge')
    ACCURACY = evaluate.load('accuracy')

    def __init__(self, tokenizer, decoder_name: str = "gpt2"):
        self.tokenizer = tokenizer
        self.decoder_id = decoder_name

    @classmethod
    def n_grams(cls, text: str, n: int) -> list:
        words = text.split(" ")
        return [tuple([words[x] for x in range(i, i+n)]) for i in range(len(words) - n + 1)]

    @classmethod
    def distinct(cls, seqs):
        """
        source:
        https://github.com/PaddlePaddle/models/blob/release/1.6/PaddleNLP/Research/Dialogue-PLATO/plato/metrics/metrics.py
        :param seqs: seq of token sequence
        :return:
        """
        """ Calculate intra/inter distinct 1/2. """
        batch_size = len(seqs)
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all = Counter(), Counter()
        for seq in seqs:
            if isinstance(seq, str):
                seq = seq.split()
            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))
            intra_dist1.append((len(unigrams) + 1e-12) / (len(seq) + 1e-5))
            intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq) - 1) + 1e-5))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
        inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        return {'intra_dist1': intra_dist1,
                'intra_dist2': intra_dist2,
                'inter_dist1': inter_dist1,
                'inter_dist2': inter_dist2}

    def compute(self, pred: EvalPrediction) -> dict:
        result = dict()

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # metrics using integer
        # result.update(self.ACCURACY.compute(references=labels_ids, predictions=pred_ids))

        # convert ids to token ids version
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # metrics using token version
        result.update(self.ROUGE.compute(predictions=pred_str, references=label_str))
        result.update(self.PPL.compute(predictions=pred_str, model_id=self.decoder_id))
        result['bluert_score'] = self.BLEURT.compute(predictions=pred_str, references=label_str)['scores']
        result.update(self.distinct(seqs=pred_str))

        return result
