# Neural Auctions
A centralized repository for neural auction projects. Developed collaboratively by Alex Stein, Avi Schwarzschild, and Michael Curry, all at the University of Maryland. This repository contains an implementation of RegretNet ([Dütting et al.](https://arxiv.org/abs/1706.03459)) with scripts to train and test them. Additionally, this repository houses code to attack neural auctions with the goal of recovering private bidder information at test time. The attack code (with defense options) is available in [invert_model.py](invert_model.py). Our work on the topic is available in our paper called [Protecting Bidder Information in Neural Auctions](https://openreview.net/pdf?id=b5RD94lXu2j).

## Getting Started

### Requirements
This code was developed and tested with Python 3.10.4.

To install requirements:

```
$ pip install -r requirements.txt
```

## Training 

To train models, run [train_model.py](train_model.py) with the desired command line arguments. With these arguments, you can modify the model architecture and set hyperparameters. The default values for all the arguments in the [hydra](config) directory configuration files and will work together to train a RegretNet. To try this, run the following.

```$ python train_model.py```

This command will train and save a model. For more examples see the [launch](launch) directory, where we have more examples.

## Saving Protocol

Each time [train_model.py](train_model.py) is executed, a hash-like adjective-Name combiniation is created and saved as the `run_id` for that execution. The `run_id` is used to save checkpoints and results without being able to accidentally overwrite any previous runs with similar hyperparameters. The folder used for saving both checkpoints and results can be chosen using the following command line argument.

```$ python train_model.py experiment_name=<path_to_exp>```

During training, models are saved periodically in the folder, when the final checkpoint at `outputs/train/<experiment_name>/<run_id>/model_final.pth` and the corresponding arguments for that run are saved in `outputs/train/<experiment_name>/<run_id>/.hydra/`. The `<experiment_name>/<run_id>/` string is necessary to later run the [test_model.py](test_model.py) and [invert_model.py](invert_model.py) files for testing and inverting these checkpoints.

The results (i.e. accuracy metrics) for the test data used in the [train_model.py](train_mode.py) run are saved in `outputs/train/<experiment_name>/<run_id>/stats.json`, the tensorboard data is saved in `outputs/train/<experiment_name>/<run_id>/tensorboard`.

The outputs directory should be as follows. Note that the default value of `<experiment_name>` is `default`, and that `neat-Chrishawn` is the adjective-Name combination for this example.
```
outputs
└── train
    ├── default
    │     └── neat-Chrishawn
    │         ├── .hydra
    │         │     ├── config.yaml
    │         │     ├── hydra.yaml
    │         │     └── overrides.yaml
    │         ├── model_final.pth
    │         ├── result.json
    │         ├── tensorboard-neat-Chrishawn-None
    │         │     └── events.out
    │         └── train.log


```

## Testing

To test a saved model, run [test_model.py](test_model.py) as follows. 

```$ python test_model.py load.model_path=<path to checkpoint>```

To point to the command line arguments that were used during training and to the model checkpoint file, use the flags in the example above. Other command line arguments are outlined in the code itself, and generally match the structure used for training. As with training, the `outputs` folder will have performance metrics in json data. (See the saving protocol below.)

## Saving Protocol (during testing)

For testing, you can run the following commandline argument to specify the location of the outputs.

```$ python test_model.py experiment_name=<experiment_name>```

This creates another unique `run_id` adjective-Name combination (different from the one created during training) and the results are saved in `outputs/test/<experiment_name>/<run_id>/<train time run_id>_result.json`.

## Analysis

To generate a pivot table with average accuracies over several trials, [table_of_training_results.py](table_of_training_results.py), [table_of_testing_results.py](table_of_testing_results.py), and [table_of_inverting_results.py](table_of_inverting_results.py) are helpful. The first command line argument (without a flag) points to an ouput directory. All the json results are then read in and averages over similar runs are nicely tabulated. For example, if you run a few trials of `train_model.py` with the same command line arguments, including `experiment_name=my_experiment`, then you can run 

```$ python table_of_training_results.py outputs/train/my_experiment```

to see the results in an easy-to-read format.

## Development and Contribution

This section is a sort-of to-do list. We are planning to get to these tasks but we also welcome contribution from the community.

- Add other neural architectures for auction mechanisms.
- Add support for other distributions of bidder profiles.

*We believe in open-source community driven software development. Please open issues and pull requests with any questions or improvements you have.*

