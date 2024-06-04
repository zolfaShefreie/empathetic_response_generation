from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from typing import Optional
import torch
import abc


class BaseMultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class MultiTaskModel(PreTrainedModel, abc.ABC):

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

    def forward(self, **kwargs) -> BaseMultiTaskOutput:
        """
        model.forward()
        run .forward() for each task and return their output
        WARNING: don't use positional argument in you code
        :param kwargs:
        :return: a object of class of with attribute of lost and output for each task
        """
        output_obj = BaseMultiTaskOutput()
        losses = list()

        for task_id, task_model in self.TASK_CONFIG.items():
            # get input data for forward function for each task
            task_kwarg = {task_arg_name: kwargs.get(arg_name, None)
                          for arg_name, task_arg_name in self.FORWARD_ARGUMENT_CONFIG[task_id].items()}
            # run .forward for task and get output
            task_output = task_model.forward(return_dict=True, **task_kwarg)

            # add its loss to list
            if task_output.loss is not None:
                losses.append(task_output.loss)
            # add attribute and its value of one task to output object
            setattr(output_obj, task_id, task_output)

        # aggregate losses
        output_obj.loss = sum(losses) if len(losses) > 0 else None

        return output_obj
