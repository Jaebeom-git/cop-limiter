import argparse
import nimblephysics_libs.biomechanics

from cli.train import TrainCommand
from cli.visualize import VisualizeCommand
from cli.create_splits import CreateSplitsCommand
from cli.inference import InferenceCommand
from cli.eval_model import EvaluateCommand
import nimblephysics as nimble
import logging

def main(args=None):
    commands = [TrainCommand(),
                VisualizeCommand(),
                CreateSplitsCommand(),
                InferenceCommand(),
                EvaluateCommand()
                ]

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='InferBiomechanics Command Line Interface')

    # Split up by command
    subparsers = parser.add_subparsers(dest="command")

    # Add a parser for each command
    for command in commands:
        command.register_subcommand(subparsers)

    # If args is None, use default parsing (from sys.argv)
    if args is None:
        args = parser.parse_args()  # Use command-line arguments
    else:
        args = parser.parse_args(args)  # Use passed arguments list

    for command in commands:
        if command.run(args):
            return


if __name__ == '__main__':
    # logpath = "log"
    # # Create and configure logger
    # logging.basicConfig(filename=logpath,
    #                     format='%(asctime)s %(message)s',
    #                     filemode='a')

    # # Creating an object
    # logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    # # Setting the threshold of logger to INFO
    # logger.setLevel(logging.INFO)

    main()  # Default behavior (command line arguments)
