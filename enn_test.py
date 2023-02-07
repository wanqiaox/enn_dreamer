from acme.utils.loggers.terminal import TerminalLogger

from enn import losses
from enn import networks
from enn import supervised
from enn.supervised import regression_data
import optax

# A small dummy dataset
dataset = regression_data.make_dataset()

# Logger
logger = TerminalLogger('supervised_regression')

# ENN
enn = networks.MLPEnsembleMatchedPrior(
    output_sizes=[50, 50, 1],
    num_ensemble=10,
)

# Loss
loss_fn = losses.average_single_index_loss(
    single_loss=losses.L2LossWithBootstrap(),
    num_index_samples=10
)

# Optimizer
optimizer = optax.adam(1e-3)

# Train the experiment
experiment = supervised.Experiment(
    enn, loss_fn, optimizer, dataset, seed=0, logger=logger)
experiment.train(FLAGS.num_batch)