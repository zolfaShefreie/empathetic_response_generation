import argparse
from abc import ABC, abstractmethod


class BaseInterface(ABC):

    DESCRIPTION = str()

    # keys are the name of arguments that it must be unique
    ARGUMENTS = {
        'argument_name': {
            'help': '',
            'required': True
        },
    }

    def __init__(self):
        """
        initial of interface
        """
        self.parser = argparse.ArgumentParser(description=self.DESCRIPTION)
        self.add_arguments()

        # define attribute based on arguments
        for argument in self.ARGUMENTS.keys():
            setattr(self, argument, None)

    def add_arguments(self):
        """
        add self.arguments to self.parser
        :return:
        """
        for argument_name, options in self.ARGUMENTS.items():
            self.parser.add_argument(f"--{argument_name}", **options)

    def validate_and_setter(self, obj: argparse.Namespace):
        """
        validate the arguments and set self.attr this values
        WARNING: should implement validate_argument in this class for each argument and
                 validate function must return the value (or apply some changes to it)
        :param obj: object of class
        :return:
        """
        arg_name_list = list(self.ARGUMENTS.keys())
        for arg_name in arg_name_list:
            # get value
            value = getattr(obj, arg_name, None)

            # get validate function for this argument
            validate_func = getattr(self, f"validate_{arg_name}", None)
            # validate value and get changed value
            value = validate_func(value) if validate_func is not None else value

            # setter for self attributes
            setattr(self, arg_name, value)

    def run(self):
        """
        run interface
        :return:
        """
        args = self.parser.parse_args()
        self.validate_and_setter(obj=args)
        self._run_main_process()

    @abstractmethod
    def _run_main_process(self):
        raise NotImplementedError
