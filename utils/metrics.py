import enum
from transformers import EvalPrediction, RobertaModel, RobertaTokenizer
from collections import Counter
import evaluate
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

from settings import EMPATHY_CLASSIFIER_MODELS_PATH, DYNAEVAL_MODEL_PATH, DYNAEVAL_ROBERTA_DIR
from utils import dgcn
from utils.models import T5EncoderClassifier


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
        copy_labels = np.copy(labels)
        copy_labels[copy_labels == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(copy_labels, skip_special_tokens=True)
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

        if len(self.task_list) == 1:
            func_task = getattr(self, f"compute_{self.task_list[0]}_metric", None)
            if func_task is not None:
                result.update(func_task(pred=pred.predictions, labels=pred.label_ids))

        else:
            for i, task_name in enumerate(self.task_list):
                pred_task, labels_task = pred.predictions[i], pred.label_ids[i]
                func_task = getattr(self, f"compute_{task_name}_metric", None)
                if func_task is not None:
                    result.update(func_task(pred=pred_task, labels=labels_task))

        return result


class EmpathyEvaluation:

    def __init__(self):
        self.empathy_classifier_model1 = T5EncoderClassifier(size="base",
                                                             base_context_encoder_name="roberta-base",
                                                             base_target_encoder_name='microsoft/DialoGPT-small',
                                                             num_labels=2, strategy=0)
        self.empathy_classifier_model1.load_state_dict(
            torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619600015/model.pt",
                       map_location=torch.device('cpu') if not torch.cuda.is_available()
                       else torch.device("cuda")))
        for param in self.empathy_classifier_model1.parameters():
            param.requires_grad = False

        self.empathy_classifier_model2 = T5EncoderClassifier(size="base",
                                                             base_context_encoder_name="roberta-base",
                                                             base_target_encoder_name='microsoft/DialoGPT-small',
                                                             num_labels=2, strategy=0)
        self.empathy_classifier_model2.load_state_dict(
            torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619600805/model.pt",
                       map_location=torch.device(
                           'cpu') if not torch.cuda.is_available()
                       else torch.device("cuda")))
        for param in self.empathy_classifier_model2.parameters():
            param.requires_grad = False

        self.empathy_classifier_model3 = T5EncoderClassifier(size="base",
                                                             base_context_encoder_name="roberta-base",
                                                             base_target_encoder_name='microsoft/DialoGPT-small',
                                                             num_labels=2, strategy=0)
        self.empathy_classifier_model3.load_state_dict(
            torch.load(f"{EMPATHY_CLASSIFIER_MODELS_PATH}/saved/empathy/1619601340/model.pt",
                       map_location=torch.device(
                           'cpu') if not torch.cuda.is_available()
                       else torch.device("cuda")))
        for param in self.empathy_classifier_model3.parameters():
            param.requires_grad = False

    def evaluate(self, history_conversation: list, response: str):
        logits = self.empathy_classifier_model1(context=[' '.join(history_conversation)], response=[response])
        empathy_label_1 = torch.argmax(torch.nn.functional.softmax(logits), dim=-1).tolist()[0]

        logits = self.empathy_classifier_model2(context=[' '.join(history_conversation)], response=[response])
        empathy_label_2 = torch.argmax(torch.nn.functional.softmax(logits), dim=-1).tolist()[0]

        logits = self.empathy_classifier_model3(context=[' '.join(history_conversation)], response=[response])
        empathy_label_3 = torch.argmax(torch.nn.functional.softmax(logits), dim=-1).tolist()[0]

        return {'empathy_label_1': empathy_label_1,
                'empathy_label_2': empathy_label_2,
                'empathy_label_3': empathy_label_3,
                'empathy': empathy_label_1 == 1 or empathy_label_2 == 1 or empathy_label_3 == 1}

    def evaluate_batch(self, history_conversations: list, responses: list):
        logits = self.empathy_classifier_model1(context=[' '.join(each) for each in history_conversations],
                                                response=responses)
        empathy_label_1 = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1).tolist()

        logits = self.empathy_classifier_model2(context=[' '.join(each) for each in history_conversations],
                                                response=responses)
        empathy_label_2 = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1).tolist()

        logits = self.empathy_classifier_model3(context=[' '.join(each) for each in history_conversations],
                                                response=responses)
        empathy_label_3 = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1).tolist()

        empathy = list()
        for idx in range(len(empathy_label_1)):
            empathy.append(empathy_label_1[idx] == 1 or empathy_label_2[idx] == 1 or empathy_label_3[idx] == 1)

        return {'empathy_label_1': empathy_label_1,
                'empathy_label_2': empathy_label_2,
                'empathy_label_3': empathy_label_3,
                'empathy': empathy}


class FedMetric:
    """source of code https://github.com/exe1023/DialEvalMetrics/blob/main/usr_fed/fed/fed.py"""

    def __init__(self, name: str = "microsoft/DialoGPT-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelWithLMHead.from_pretrained(name)

    def score_batch(self, texts, batch_size=-1, max_seq_length=256):
        """
        :param texts: list of string
        :param batch_size: specify the batch size you want to use in inference. -1 means packing all queries in 1 batch.
        :param max_seq_length: specify the maximum sequence length after tokenization. Max: 1024
        :return:
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # make sure all text will in 1024:
        text_batchs = []
        for text in texts:
            tokenized = self.tokenizer.tokenize(text)
            if len(tokenized) > max_seq_length:
                tokenized = tokenized[-(max_seq_length):]
                tokenized[0] = self.tokenizer.eos_token  # max sure we have special token at beginning.
            text_batchs.append(tokenized)

        # pad the input and generate attention mask
        pad_idx = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])
        token_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in text_batchs]
        max_text_length = max([len(s) for s in token_ids])
        padded_tokens = [tok_ids + (pad_idx * (max_text_length - len(tok_ids))) for tok_ids in token_ids]
        input_ids = torch.tensor(padded_tokens)
        attention_mask = torch.zeros(input_ids.shape).long()
        for idx, tok_ids in enumerate(token_ids):
            attention_mask[idx][:len(tok_ids)] = 1

        model = self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            if batch_size == -1:
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs[1]
            else:
                logits = []
                for i in range(0, input_ids.size(0), batch_size):
                    outputs = model(input_ids[i:i + batch_size, :],
                                    attention_mask=attention_mask[i:i + batch_size, :],
                                    labels=input_ids[i:i + batch_size, :])
                    logits.append(outputs[1])
                logits = torch.cat(logits, dim=0)
        shifted_logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(shifted_logits.view(-1, model.config.vocab_size), labels.view(-1))

        return lm_loss.view(len(texts), -1)

    def score(self, text):
        """
        compute score
        :param text:
        :return:
        """
        if not text.startswith("<|endoftext|> "):
            text = "<|endoftext|> " + text
        # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        tokenize_input = self.tokenizer.tokenize(text)

        if len(tokenize_input) >= 256:
            tokenize_input = ['<|endoftext|>'] + tokenize_input[-256:]
        # 50256 is the token_id for <|endoftext|>
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        with torch.no_grad():
            outputs = self.model(tensor_input, labels=tensor_input)
            loss, logits = outputs[:2]
        return loss.item()

    @staticmethod
    def prepare_inputs(history_conversation: list, response: str):
        """
        preprocess of input
        :param history_conversation:
        :param response:
        :return:
        """
        return '<|endoftext|> ' + ' <|endoftext|> '.join(history_conversation) + ' <|endoftext|> ' + response

    def evaluate(self, history_conversation: list, response: str, max_batch_size=16, max_seq_length=400):
        """
        compute fed metric
        :param history_conversation:
        :param response:
        :param max_batch_size:
        :param max_seq_length:
        :return:
        """
        conversation = self.prepare_inputs(history_conversation=history_conversation,
                                           response=response)
        scores = {}
        turn_level_utts = {
            "interesting": {
                "positive": ["Wow that is really interesting.", "That's really interesting!",
                             "Cool! That sounds super interesting."],
                "negative": ["That's not very interesting.", "That's really boring.",
                             "That was a really boring response."]
            },
            "engaging": {
                "positive": ["Wow! That's really cool!", "Tell me more!",
                             "I'm really interested in learning more about this."],
                "negative": ["Let's change the topic.", "I don't really care. That's pretty boring.",
                             "I want to talk about something else."]
            },
            "specific": {
                "positive": ["That's good to know. Cool!", "I see, that's interesting.", "That's a good point."],
                "negative": ["That's a very generic response.", "Not really relevant here.",
                             "That's not really relevant here."]
            },
            "relevant": {
                "positive": [],
                "negative": ["That's not even related to what I said.", "Don't change the topic!",
                             "Why are you changing the topic?"]
            },
            "correct": {
                "positive": [],
                "negative": ["You're not understanding me!", "I am so confused right now!",
                             "I don't understand what you're saying."]
            },
            "semantically appropriate": {
                "positive": ["That makes sense!", "You have a good point."],
                "negative": ["That makes no sense!"]
            },
            "understandable": {
                "positive": ["That makes sense!", "You have a good point."],
                "negative": ["I don't understand at all!", "I'm so confused!", "That makes no sense!",
                             "What does that even mean?"]
            },
            "fluent": {
                "positive": ["That makes sense!", "You have a good point."],
                "negative": ["Is that real English?", "I'm so confused right now!", "That makes no sense!"]
            },
        }

        texts = list()
        for metric, utts in turn_level_utts.items():
            pos, neg = utts["positive"], utts['negative']
            for m in pos:
                texts.append(conversation + " <|endoftext|> " + m)
            for m in neg:
                texts.append(conversation + " <|endoftext|> " + m)

        loss = self.score_batch(texts, batch_size=max_batch_size, max_seq_length=max_seq_length)

        idx = 0
        for metric, utts in turn_level_utts.items():
            pos, neg = utts["positive"], utts['negative']
            if len(pos) > 0:
                high_score = loss[idx: idx + len(pos), :].mean().item()
            else:
                high_score = 0
            idx += len(pos)
            if len(neg) > 0:
                low_score = loss[idx: idx + len(neg), :].mean().item()
            else:
                low_score = 0
            idx += len(neg)
            scores[metric] = (low_score - high_score)

        dialog_level_utts = {
            "coherent": {
                "positive": [],
                "negative": ["You're making no sense at all.", "You're changing the topic so much!",
                             "You are so confusing."]
            },
            "error recovery": {
                "positive": [],
                "negative": ["I am so confused right now.", "You're really confusing.",
                             "I don't understand what you're saying."]
            },
            "consistent": {
                "positive": [],
                "negative": ["That's not what you said earlier!", "Stop contradicting yourself!"],
            },
            "diverse": {
                "positive": [],
                "negative": ["Stop saying the same thing repeatedly.", "Why are you repeating yourself?",
                             "Stop repeating yourself!"]
            },
            "depth": {
                "positive": [],
                "negative": ["Stop changing the topic so much.", "Don't change the topic!"],
            },
            "likeable": {
                "positive": ["I like you!", "You're super polite and fun to talk to", "Great talking to you."],
                "negative": ["You're not very nice.", "You're not very fun to talk to.", "I don't like you."]
            },
            "understand": {
                "positive": [],
                "negative": ["You're not understanding me!", "What are you trying to say?",
                             "I don't understand what you're saying."]
            },
            "flexible": {
                "positive": ["You're very easy to talk to!", "Wow you can talk about a lot of things!"],
                "negative": ["I don't want to talk about that!", "Do you know how to talk about something else?"],
            },
            "informative": {
                "positive": ["Thanks for all the information!", "Wow that's a lot of information.",
                             "You know a lot of facts!"],
                "negative": ["You're really boring.", "You don't really know much."],
            },
            "inquisitive": {
                "positive": ["You ask a lot of questions!", "That's a lot of questions!"],
                "negative": ["You don't ask many questions.", "You don't seem interested."],
            },
        }

        texts = list()
        for metric, utts in dialog_level_utts.items():
            pos, neg = utts["positive"], utts['negative']
            for m in pos:
                texts.append(conversation + " <|endoftext|> " + m)
            for m in neg:
                texts.append(conversation + " <|endoftext|> " + m)
        loss = self.score_batch(texts, batch_size=max_batch_size, max_seq_length=max_seq_length)
        idx = 0
        for metric, utts in dialog_level_utts.items():
            pos, neg = utts["positive"], utts['negative']
            if len(pos) > 0:
                high_score = loss[idx: idx + len(pos), :].mean().item()
            else:
                high_score = 0
            idx += len(pos)
            if len(neg) > 0:
                low_score = loss[idx: idx + len(neg), :].mean().item()
            else:
                low_score = 0
            idx += len(neg)
            scores[metric] = (low_score - high_score)

        return scores


class DynaEvalMetric:

    class Args:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 1
        learning_rate = 0.002
        weight_decay = 1e-8
        max_grad_value = -1
        drop_rate = 0.5
        wp = 4
        wf = 4
        n_speakers = 2
        hidden_size = 100
        rnn = "lstm"
        class_weight = "store_true"
        sentence_dim = 768
        max_seq_len = 50
        max_dialogue_len = 700
        seed = 24

    def __init__(self, batch_size: int):
        self.args = DynaEvalMetric.Args()
        self.args.batch_size = batch_size
        model_file = DYNAEVAL_MODEL_PATH
        self.model = dgcn.DynaEval(self.args).to(self.args.device)

        ckpt = torch.load(model_file, map_location=torch.device(self.args.device))
        best_dev_f1 = ckpt["best_dev_acc"]
        best_epoch = ckpt["best_epoch"]
        best_state = ckpt["best_state"]
        self.model.load_state_dict(best_state, strict=False)
        self.model.eval()

        self.sentence_model = RobertaModel.from_pretrained(DYNAEVAL_ROBERTA_DIR).to(self.args.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(DYNAEVAL_ROBERTA_DIR)

    def preprocess(self, history_conversation: list, response: str, idx=0):
        utts_1 = history_conversation + [response]
        utts_2 = history_conversation + [response]
        spk_1_list = ['A' if j % 2 == 0 else 'B' for j in range(len(utts_1))]
        spk_2_list = ['A' if j % 2 == 0 else 'B' for j in range(len(utts_2))]

        sample = dgcn.Sample(vid="eval_{}".format(idx),
                             speaker_1=spk_1_list,
                             speaker_2=spk_2_list,
                             text_1=utts_1,
                             text_2=utts_2,
                             label=1)
        return sample

    def evaluate(self, history_conversation: list, response: str):
        evalset = dgcn.Dataset(samples=[self.preprocess(history_conversation=history_conversation, response=response)],
                            model=self.sentence_model,
                            tokenizer=self.tokenizer,
                            args=self.args)

        with torch.no_grad():
            preds = []
            for idx in range(len(evalset)):
                data = evalset[idx]
                data = {k: v.to(self.args.device) if 'len' not in k else v for k, v in data.items()}
                rst = self.model(data)
                scores = rst[1]
                preds.append(scores.detach().to("cpu"))

            preds = torch.nn.functional.sigmoid(torch.cat(preds, dim=-1)).tolist()
            return {'dynaeval_score': preds[0]}

    def evaluate_batch(self, history_conversations: list, responses: list):
        samples = [self.preprocess(history_conversation=history_conversations[idx], response=responses[idx])
                   for idx in range(len(responses))]
        evalset = dgcn.Dataset(samples=samples,
                               model=self.sentence_model,
                               tokenizer=self.tokenizer,
                               args=self.args)

        with torch.no_grad():
            preds = []
            for idx in range(len(evalset)):
                data = evalset[idx]
                data = {k: v.to(self.args.device) if 'len' not in k else v for k, v in data.items()}
                rst = self.model(data)
                scores = rst[1]
                preds.append(scores.detach().to("cpu"))

            preds = torch.nn.functional.sigmoid(torch.cat(preds, dim=-1)).tolist()
            return {'dynaeval_score': preds}


class ExtraMetricsManagement:

    batch_size = 16

    @classmethod
    def run_empathy_metric(cls, test_data, history_key_name, label_key_name, generated_res_key_name):
        empathy_metric = EmpathyEvaluation()
        result_plus_data = list()
        empathy_present = 0
        batch_size = cls.batch_size
        for i in tqdm(range(int(len(test_data)/batch_size)+1), desc='running empathy metrics'):
            history_conversations = [record[history_key_name] for record in test_data[i*batch_size: (i+1)*batch_size]]
            responses = [record[generated_res_key_name] for record in test_data[i*batch_size: (i+1)*batch_size]]

            empathy_result = empathy_metric.evaluate_batch(history_conversations=history_conversations,
                                                           responses=responses)
            for index, record in enumerate(test_data[i*batch_size: (i+1)*batch_size]):
                empathy_result_record = dict()
                for k, v in empathy_result.items():
                    empathy_result_record[k] = v[index]
                result_plus_data.append({**record, **empathy_result_record})

            for each in empathy_result['empathy']:
                if each:
                    empathy_present += 1

        return result_plus_data, {'empathy_present': empathy_present / len(test_data)}

        # for record in tqdm(test_data, desc='running empathy metrics'):
        #     empathy_result = empathy_metric.evaluate(history_conversation=record[history_key_name],
        #                                              response=record[generated_res_key_name])
        #     result_plus_data.append({**record, **empathy_result})
        #
        #     if empathy_result['empathy']:
        #         empathy_present += 1
        # return result_plus_data, {'empathy_present': empathy_present/len(test_data)}

    @classmethod
    def run_fed_metric(cls, test_data, history_key_name, label_key_name, generated_res_key_name):
        fed_metric = FedMetric()
        result_plus_data = list()
        metrics = dict()
        for record in tqdm(test_data, desc='running FED'):
            fed_result = fed_metric.evaluate(history_conversation=record[history_key_name],
                                             response=record[generated_res_key_name], max_batch_size=cls.batch_size)
            result_plus_data.append({**record, **fed_result})

            for key, value in fed_result.items():
                metrics[key] = metrics.get(key, 0) + value

        return result_plus_data, {key: value/len(test_data) for key, value in metrics.items()}

    @classmethod
    def run_dynaeval_metrics(cls, test_data, history_key_name, label_key_name, generated_res_key_name):
        dynaeval = DynaEvalMetric(batch_size=8)
        result_plus_data = list()
        metrics = dict()
        batch_size = cls.batch_size
        for i in tqdm(range(int(len(test_data) / batch_size) + 1), desc='running empathy metrics'):
            history_conversations = [record[history_key_name] for record in test_data[i * batch_size: (i + 1) * batch_size]]
            responses = [record[generated_res_key_name] for record in test_data[i * batch_size: (i + 1) * batch_size]]

            dunaeval_result = dynaeval.evaluate_batch(history_conversations=history_conversations,
                                                      responses=responses)
            for index, record in enumerate(test_data[i * batch_size: (i + 1) * batch_size]):
                dynaeval_result_record = dict()
                for k, v in dunaeval_result.items():
                    dynaeval_result_record[k] = v[index]
                    metrics[k] = metrics.get(k, 0) + v[index]
                result_plus_data.append({**record, **dynaeval_result_record})

        return result_plus_data, {key: value / len(test_data) for key, value in metrics.items()}

        # for record in tqdm(test_data, desc='running dynaEval'):
        #     dunaeval_result = dynaeval.evaluate(history_conversation=record[history_key_name],
        #                                      response=record[generated_res_key_name])
        #     result_plus_data.append({**record, **dunaeval_result})
        #
        #     for key, value in dunaeval_result.items():
        #         metrics[key] = metrics.get(key, 0) + value
        # return result_plus_data, {key: value / len(test_data) for key, value in metrics.items()}

    @classmethod
    def compute(cls, test_data, history_key_name, label_key_name, generated_res_key_name, include_fed=True,
                include_empathy=True, include_dynaeval=True):
        metric_result = dict()
        if include_empathy:
            test_data, metrics = cls.run_empathy_metric(test_data=test_data, history_key_name=history_key_name,
                                                        label_key_name=label_key_name,
                                                        generated_res_key_name=generated_res_key_name)
            metric_result.update(metrics)

        if include_fed:
            test_data, metrics = cls.run_fed_metric(test_data=test_data, history_key_name=history_key_name,
                                                    label_key_name=label_key_name,
                                                    generated_res_key_name=generated_res_key_name)
            metric_result.update(metrics)

        if include_dynaeval:
            test_data, metrics = cls.run_dynaeval_metrics(test_data=test_data, history_key_name=history_key_name,
                                                          label_key_name=label_key_name,
                                                          generated_res_key_name=generated_res_key_name)
            metric_result.update(metrics)

        return test_data, metric_result

