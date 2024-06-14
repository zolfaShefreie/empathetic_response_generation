from typing import Optional, List, Dict, Union, Any, Tuple, Callable
from datasets import Dataset
from transformers import Seq2SeqTrainer, EncoderDecoderModel, GenerationConfig
import torch.nn as nn
import torch
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
  from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

from utils.models import MultiTaskModel


class MultiTaskTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        model: Union[MultiTaskModel, nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.list_generation_model = [task_id for task_id, task_model in model.TASK_CONFIG.items()
                                      if isinstance(task_model, EncoderDecoderModel)]

    def prediction_step(
        self,
        model: MultiTaskModel,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.
        :param model:
        :param inputs: The inputs and targets of the model.
        :param prediction_loss_only: Whether or not to return the loss only.
        :param ignore_keys:
        :param gen_kwargs: Additional `generate` specific kwargs.
        :return: loss, logits, labels
        """

        has_labels = self._has_labels(inputs)

        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        if has_labels or loss_without_labels:
            labels = nested_detach({name: inputs.get(name) for name in self.label_names})
            # if len(labels) == 1:
            #     labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            generated_tokens = None
            if model.get_generative_task_id() is not None and self.args.predict_with_generate and not prediction_loss_only:
                generated_tokens = self.generate_in_prediction_step(model, inputs)

            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    loss_mb = raw_outputs["loss"]
                    loss = loss_mb.reduce_mean().detach().cpu()

                else:
                    loss = None

                if generated_tokens is not None:
                    raw_outputs[model.get_generative_output_key_name()] = generated_tokens
                logits_mb = {k: v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"]}
                logits = smp_nested_concat(logits_mb)

            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if generated_tokens is not None:
                        outputs[model.get_generative_output_key_name()] = generated_tokens

                    logits = {k: v for k, v in outputs.items()
                              if ('logit' in k) and (k not in ignore_keys + ["loss"])}

                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)

                    if generated_tokens is not None:
                        outputs[model.get_generative_output_key_name()] = generated_tokens

                    logits = {k: v for k, v in outputs.items() if ('logit' in k) and (k not in ignore_keys)}

                    # TODO: this used for xl transformers and i don't know how to handle it in multitask learning
                    # if self.args.past_index >= 0:
                    #     self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        # make a function for it
        if has_labels and model.get_generative_task_id() is not None:
            key_name = {v: k for k, v in model.FORWARD_ARGUMENT_CONFIG[model.get_generative_task_id()].items()}['labels']
            generative_task_labels = inputs[key_name]
            if generative_task_labels.shape[-1] < self.model.generation_config.max_length:
                generative_task_labels = self._pad_tensors_to_max_len(generative_task_labels, self.model.generation_config.max_length)
            elif self.model.generation_config.max_new_tokens is not None and generative_task_labels.shape[-1] < self.model.generation_config.max_new_tokens + 1:
                generative_task_labels = self._pad_tensors_to_max_len(labels, self.model.generation_config.max_new_tokens + 1)

            labels[key_name] = generative_task_labels

        logits = nested_detach(logits)

        # change orders
        logits, labels = self._put_to_correct_order_tuple(model, logits, labels)

        if len(logits) == 1:
            logits = logits[0]

        if labels is not None and len(labels) == 1:
            labels = labels[0]

        return loss, logits, labels

    def _put_to_correct_order_tuple(self, model: MultiTaskModel, logits: dict, labels: dict = None, ) -> tuple:
        """
        change the order of labels and logits to model.task_order and convert to tuple
        :param model: The model to evaluate.
        :param logits: the output of model
        :param labels: the labels of inputs
        :return: logits, labels
        """

        if labels is None:
            result = list()
            for task_id in model.TASK_ORDER:
                result += [v for k, v in logits.items() if task_id in k]
            return tuple(result), None

        else:
            label_task_mapping = dict()
            for task_k, task_forward_arg_config in model.FORWARD_ARGUMENT_CONFIG.items():
                label_task_mapping[task_k] = [pass_key_name for pass_key_name, o_key_name in task_forward_arg_config.items()
                                              if pass_key_name in labels.keys()]

            output_task_mapping = dict()
            for task_id in model.TASK_ORDER:
                output_task_mapping[task_id] = [k for k, v in logits.items() if task_id in k]
            if model.get_generative_task_id() is not None:
                output_task_mapping[model.get_generative_task_id()] = [model.get_generative_output_key_name(), ]

            result_labels, result_logits = list(), list()
            for task_id in model.TASK_ORDER:
                result_logits += [logits[each] for each in sorted(output_task_mapping[task_id])]
                result_labels += [labels[each] for each in sorted(label_task_mapping[task_id])]
            return tuple(result_logits), tuple(result_labels)

    def _has_labels(self,
                    inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        check if one label is in the inputs
        WARNING: two option are considered in MultiTaskTrainer:
            1. all task have labels
            2. none of task doesn't have label
        :param inputs: The inputs and targets of the model.
        :return: a boolean
        """
        for each in inputs:
            if 'label' in each:
                return True
        return False

    def generate_in_prediction_step(self, model, inputs, **gen_kwargs):
        """
        perform model.generate() for generative task in evaluation stage
        :param model: The model to evaluate.
        :param inputs: The inputs and targets of the model.
        :param gen_kwargs: Additional `generate` specific kwargs.
        :return:
        """

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        if gen_kwargs.get("bos_token_id", None) is None:
            gen_kwargs['bos_token_id'] = model.generative_task_generation_config.bos_token_id
        if gen_kwargs.get("pad_token_id", None) is None:
            gen_kwargs['pad_token_id'] = model.generative_task_generation_config.pad_token_id
        if gen_kwargs.get("eos_token_id", None) is None:
            gen_kwargs['eos_token_id'] = model.generative_task_generation_config.eos_token_id

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()

        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        return generated_tokens
