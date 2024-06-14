import dataclasses
from transformers import PreTrainedModel, PretrainedConfig, RobertaPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional
import torch
import abc


@dataclasses.dataclass
class BaseMultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class MultiTaskModel(PreTrainedModel, abc.ABC):
    """
    WARNING:
        you can't use this class for multiple generative models.
    """

    def __init__(self,
                 config: PretrainedConfig = PretrainedConfig(),
                 *inputs,
                 **kwargs):

        super().__init__(config=config, *inputs, **kwargs)

        self.TASK_CONFIG = self.initial_models(**kwargs)
        self._validate_task_config()

        self.set_shared_layers()

        self.FORWARD_ARGUMENT_CONFIG = self.get_arg_forward_settings()
        self._validate_task_config()

        if self.get_generative_task_id() is not None and self.get_generative_task_id() not in self.TASK_CONFIG.keys():
            raise Exception(f"{self.get_generative_task_id()} not found")

        if self.get_generative_task_id() is not None:
            self.generation_config = self.TASK_CONFIG[self.get_generative_task_id()].generation_config
            self.config.is_encoder_decoder = True

        # these will be used on multiTaskTrainer
        self.TASK_ORDER = list(self.TASK_CONFIG)

    @property
    def generative_task_generation_config(self):
        return self.TASK_CONFIG[self.get_generative_task_id()].generation_config

    @abc.abstractmethod
    def get_generative_task_id(self):
        """
        get the task id of generative task. if there is no generative task return None
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_encoder(self):
        """
        get encoder of model if model is encoderdecoder model
        :return:
        """
        raise NotImplementedError

    def get_generative_output_key_name(self):
        """
        get the output key name of generative task.
        :return:
        """
        return "logits"

    @abc.abstractmethod
    def initial_models(self, **kwargs) -> dict:
        """
        initial models
        :param kwargs: the arguments that is pasted by __init__ function
        :return: a dictionary that shows task_id and its task (model)
            Example:
            {
            "task_1": self.response_generator,
            "task_2": self.classifier,
            ...
            }

            WARNING: if you doesn't put tasks correctly in dict the model doesn't run the task for training and testing
        """
        raise NotImplemented

    def _validate_task_config(self):
        """
        check everything is fine at self.TASK_CONFIG
        :return:
        """
        # check valid format of self.TASK_CONFIG
        if not isinstance(self.TASK_CONFIG, dict):
            raise TypeError("wrong format")

        # check the type of each task
        # all of task must be from HuggingFace
        for each in self.TASK_CONFIG.values():
            if not isinstance(each, PreTrainedModel):
                raise Exception("all task must be HuggingFace model")

    @abc.abstractmethod
    def set_shared_layers(self):
        """
        make the specific layers shared
        :return:
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_arg_forward_settings(self) -> dict:
        """
        get a dictionary of information about forward function's arguments based on tasks.
        this dictionary is used for passing or mapping function's inputs
        :return: dictionary with below format
        {
            "task_id": {
                "arg_name_in_forward_func": "arg_name of forward functions of task model"
            }
        }

        Example:
        {
            "task_1": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels_1": "labels",
            },

            "task_2": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels_2": "labels",
            }
        }

        """
        raise NotImplemented

    def _validate_forward_args_settings(self):
        """
        check everything is fine in self.self.FORWARD_ARGUMENT_CONFIG
        :return:
        """
        # check valid format of self.FORWARD_ARGUMENT_CONFIG
        if not isinstance(self.FORWARD_ARGUMENT_CONFIG, dict):
            raise TypeError("wrong format")

        # check self.FORWARD_ARGUMENT_CONFIG based on self.TASK_CONFIG
        if len(self.TASK_CONFIG) != len(self.FORWARD_ARGUMENT_CONFIG):
            raise Exception("task numbers not equal to task arguments settings")
        for task_id in self.TASK_CONFIG.keys():
            if task_id not in self.FORWARD_ARGUMENT_CONFIG:
                raise Exception(f"doesn't add arguments settings for {task_id} task")

        # check the arguments exist in original forward function (forward func of a task)
        for task_id, task_model in self.TASK_CONFIG.items():

            forward_arguments = getattr(task_model, "forward").__code__.co_varnames
            for argument in self.FORWARD_ARGUMENT_CONFIG[task_id]:
                if argument not in forward_arguments:
                    raise Exception(f"invalid argument in {task_id} task")

    def get_output_single_task(self, task_id, is_generative: bool, **data_kwargs) -> (dict, float):
        """
        run forward function for single task
        :param task_id: task_id
        :param is_generative: a boolean that shows whether task is generative
        :param data_kwargs:
        :return: output, loss
        """
        output_dict = dict()
        # get input data for forward function for each task
        task_kwarg = {task_arg_name: data_kwargs.get(arg_name, None)
                      for arg_name, task_arg_name in self.FORWARD_ARGUMENT_CONFIG[task_id].items()}
        # run .forward for task and get output
        task_output = self.TASK_CONFIG[task_id].forward(return_dict=True, **task_kwarg)

        # add attribute and its value of one task to output object
        if is_generative:
            # use the original names for generative task to avoid overriding .generate() function
            for attr_name, attr_value in task_output.__dict__.items():
                if attr_name == "loss":
                    output_dict[f"{task_id}_{attr_name}"] = attr_value
                else:
                    output_dict[f"{attr_name}"] = attr_value

        else:
            for attr_name, attr_value in task_output.__dict__.items():
                output_dict[f"{task_id}_{attr_name}"] = attr_value

        return output_dict, task_output.get('loss', None)

    def forward(self, task_id=None, **kwargs) -> BaseMultiTaskOutput:
        """
        model.forward()
        run .forward() for each task and return their output
        WARNING: don't use positional argument in you code
        :param task_id:
        :param kwargs:
        :return: a object of class of with attribute of lost and output for each task
        """
        output_obj = BaseMultiTaskOutput()
        losses = list()
        if task_id is not None:
            output_dict, loss = self.get_output_single_task(task_id=task_id,
                                                            is_generative=self.get_generative_task_id() == task_id,
                                                            **kwargs)

            for k, v in output_dict.items():
                output_obj[k] = v

            # add its loss to list
            if loss is not None:
                losses.append(loss)

        else:
            for task_id, task_model in self.TASK_CONFIG.items():
                output_dict, loss = self.get_output_single_task(task_id=task_id,
                                                                is_generative=self.get_generative_task_id() == task_id,
                                                                **kwargs)

                for k, v in output_dict.items():
                    output_obj[k] = v

                # add its loss to list
                if loss is not None:
                    losses.append(loss)

        # aggregate losses
        output_obj['loss'] = sum(losses) if len(losses) > 0 else None

        return output_obj

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        if self.get_generative_task_id() is not None:
            result = self.TASK_CONFIG[self.get_generative_task_id()].prepare_inputs_for_generation(*args, **kwargs)
            result.update({'task_id': self.get_generative_task_id()})
            return result
        return None

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply encoder-decoder cache reordering here
        if self.get_generative_task_id() is not None:
            return self.TASK_CONFIG[self.get_generative_task_id()]._reorder_cache(past_key_values, beam_idx)
