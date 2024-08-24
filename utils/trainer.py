import shutil
from typing import Optional, List, Dict, Union, Any, Tuple, Callable
from datasets import Dataset
from transformers import Seq2SeqTrainer, EncoderDecoderModel, GenerationConfig, Trainer
import torch.nn as nn
import torch
import os
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import nested_detach, nested_concat
from transformers.utils.import_utils import is_peft_available
from transformers.utils import is_sagemaker_mp_enabled, is_torch_tpu_available
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
if is_sagemaker_mp_enabled():
  from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
if is_peft_available():
    from peft import PeftModel

from utils.models import MultiTaskModel


class TrainerMultiLoss(Trainer):
    """
    this trainer support logging multi loss
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.other_losses = dict()
        self.tr_total_other_losses = dict()

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None,
                             trial=None, ignore_keys_for_eval=None):
        self.tr_total_other_losses = dict()

        result = super()._inner_training_loop(batch_size=batch_size, args=args, resume_from_checkpoint=resume_from_checkpoint,
                                     trial=trial, ignore_keys_for_eval=ignore_keys_for_eval)

        self._reset_other_losses()
        return result

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
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

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True, is_eval=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, is_eval=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        self._update_other_loss_new_step(outputs, is_eval=is_eval)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _update_other_loss_new_step(self, model_output: dict, is_eval: bool = False):
        if isinstance(model_output, dict):
            prefix = 'train_phase' if not is_eval else "eval_phase"

            # trainer is in train phase
            if not is_eval:
                def preprocess_train_loss(unprocessed_loss):
                    if self.args.n_gpu > 1:
                        unprocessed_loss = unprocessed_loss.mean()
                    return unprocessed_loss.detach() / self.args.gradient_accumulation_steps

                losses_dict = {f"{prefix}_{key}": preprocess_train_loss(value)
                               for key, value in model_output.items()
                               if 'loss' in key}

                # add to get sum of loss
                self.other_losses.update({key: value + self.other_losses.get(key, 0.0)
                                          for key, value in losses_dict.items()})
            # trainer is in eval phase
            else:
                def preprocess_eval_loss(unprocessed_loss):
                    if unprocessed_loss is not None:
                        losses = self.accelerator.gather_for_metrics((unprocessed_loss.repeat(self.args.eval_batch_size)))
                        return losses
                    return None

                losses_dict = {f"{prefix}_{key}": preprocess_eval_loss(value)
                               for key, value in model_output.items()
                               if 'loss' in key}

                # concat all losses
                self.other_losses.update({key: value if self.other_losses.get(key, None) is None
                                          else nested_concat(self.other_losses.get(key, None),
                                                             value,
                                                             padding_index=-100)
                                          for key, value in losses_dict.items()})

    def _reset_other_losses(self, is_eval: bool = False):
        """
        reset other_losses
        :param is_eval:
        :return:
        """
        prefix = 'train_phase' if not is_eval else 'eval_phase'
        if not is_eval:
            self.other_losses.update({key: 0.0 for key in self.other_losses.keys() if prefix in key})
        else:
            self.other_losses.update({key: None for key in self.other_losses.keys() if prefix in key})

    def _calculate_tr_total_other_losses(self):
        """
        calculate tr_total_other_losses
        :return:
        """
        # update with last steps
        self.tr_total_other_losses.update({key: value + self.tr_total_other_losses.get(key, 0.0)
                                           for key, value in self.other_losses.items()
                                           if key in self.tr_total_other_losses})
        # get the average for each loss
        return {key: value / self.state.global_step for key, value in self.tr_total_other_losses.items()}

    def _calculate_eval_other_losses(self):
        """
        calculate eval_other_losses
        :return:
        """
        # get mean of all_losses list
        return {key: value.mean().item() for key, value in self.other_losses.items() if "eval_phase" in key}

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_other_losses = {key: self._nested_gather(value).mean().item()
                               for key, value in self.other_losses.items() if "train_phase" in key}

            # reset tr_loss to zero
            tr_loss -= tr_loss
            self._reset_other_losses(is_eval=False)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs.update({key: round(value / (self.state.global_step - self._globalstep_last_logged), 4)
                         for key, value in tr_other_losses.items()})

            self._total_loss_scalar += tr_loss_scalar
            # update tr_total_other_losses
            self.tr_total_other_losses.update({key: value + self.tr_total_other_losses.get(key, 0.0)
                                               for key, value in tr_other_losses.items()})
            print("_maybe_log_save_evaluate", self.tr_total_other_losses)
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}

        # end of train
        if "train_loss" in output:
            total_other_train_loss = self._calculate_tr_total_other_losses()
            print(total_other_train_loss)
            output.update(total_other_train_loss)
            logs.update(total_other_train_loss)
            print(logs)

        # end of eval
        if "eval_loss" in output:
            total_eval_other_loss = self._calculate_eval_other_losses()
            output.update(total_eval_other_loss)
            logs.update(total_eval_other_loss)
            self._reset_other_losses(is_eval=True)

        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class Seq2SeqTrainerMultiLoss(Seq2SeqTrainer, TrainerMultiLoss):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                    self._update_other_loss_new_step(outputs, is_eval=True)
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels


class MultiTaskTrainer(Seq2SeqTrainerMultiLoss):

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
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True, is_eval=True)
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
