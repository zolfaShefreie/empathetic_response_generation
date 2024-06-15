import enum
from transformers import EvalPrediction
from collections import Counter
import evaluate
import numpy as np
from sklearn.metrics import f1_score


class Metrics:

    class TaskEnum(enum.Enum):
        classifier = 'classifier'
        text_generator = 'text_generator'

        @classmethod
        def list(cls):
            """
            :return: list of values
            """
            return list(map(lambda c: c.value, cls))

    PPL = evaluate.load("perplexity", module_type="metric")
    # BLEURT = evaluate.load("bleurt", module_type="metric")
    ROUGE = evaluate.load('rouge')
    BLEU = evaluate.load("bleu")
    ACCURACY = evaluate.load('accuracy')
    BLEU_MAX_ORDER = 4

    def __init__(self, tokenizer, task_list: list, decoder_name: str = "gpt2"):
        """
        initial of metric object
        :param tokenizer:
        :param task_list: a list of task that shows order of outputs of tasks
        :param decoder_name: is used on PPL metric
        """
        self.tokenizer = tokenizer
        self.decoder_id = decoder_name
        self._validate_task_list(task_list=task_list)
        self.task_list = task_list

    def _validate_task_list(self, task_list: list):
        for task in task_list:
            if task not in self.TaskEnum.list():
                raise Exception("invalid task")

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

    def compute_text_generator_metric(self, pred, labels) -> dict:
        """
        compute some metrics for text generator task
        :param pred:
        :param labels:
        :return:
        """
        result = dict()

        # metrics using integer
        # result.update(self.ACCURACY.compute(references=labels_ids, predictions=pred_ids))

        # convert ids to token ids version
        pred_str = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        labels[labels == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_str = ['.' if not each else each for each in pred_str]

        # metrics using token version
        result.update(self.ROUGE.compute(predictions=pred_str, references=label_str))
        result['mean_perplexity'] = self.PPL.compute(predictions=pred_str, model_id=self.decoder_id)['mean_perplexity']
        result['bleu'] = self.BLEU.compute(predictions=pred_str, references=label_str, max_order=self.BLEU_MAX_ORDER)['bleu']
        # result['bluert_score'] = self.BLEURT.compute(predictions=pred_str, references=label_str)['scores']
        result.update(self.distinct(seqs=pred_str))

        return result

    def compute_classifier_metric(self, pred, labels) -> dict:
        """
        compute some metrics for text classifier task
        :param pred:
        :param labels:
        :return:
        """
        def after_result(z):
            result = 1/(1 + np.exp(-z))
            return np.argmax(result, axis=-1)

        pred = after_result(pred)
        result = dict()
        result['f1'] = f1_score(labels, pred, average='micro')
        result.update(self.ACCURACY.compute(references=labels, predictions=pred))
        return result

    def compute(self, pred: EvalPrediction) -> dict:
        """
        compute metrics for each task
        :param pred: output of prediction_step of trainer
        :return:
        """

        result = dict()

        for i, task_name in enumerate(self.task_list):
            pred_task, labels_task = pred.predictions[i], pred.label_ids[i]
            func_task = getattr(self, f"compute_{task_name}_metric", None)
            if func_task is not None:
                result.update(func_task(pred=pred_task, labels=labels_task))

        return result
