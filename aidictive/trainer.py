
import numpy as np

import sklearn
import sklearn.metrics

import torch

from .optimizers import create as create_optim
from .loggers import create as create_logger
from .data import get_tensor_data_loader


DEFAULT_OPTIMIZER = dict(
    name="radam"
)

DEFAULT_LOGGER = dict(
    name="interval",
    params=dict(
        epoch_interval=1,
        batch_interval=1e100,
    )
)

DEFAULT_SCHEDULER = dict(
    name="reduceonplateau",
    params=dict(
        patience=5,
    )
)

OPTIMIZER = "optimizer"
SCHEDULER = "scheduler"
TRAIN_LOGGER = "train_logger"
DL_TRAIN = "dl_train"
LOSS = "loss"
METRIC = "metric"


class Trainer(object):

    def __init__(self, model):
        # Save a reference of the model to train.
        self.model = model
        # Create an empty state.
        # An state contains all the objects needed for training.
        # Using the _settings dict we will create all the objects before
        # training.
        self._state = {}
        # Create settings with default values.
        self._settings = {}
        self.set_optimizer(DEFAULT_OPTIMIZER)
        self.set_logger(DEFAULT_LOGGER)

    def _remove_from_state(self, key):
        """Safety removes a key from the state dict.
        
        Returns True if the key was in the state, False if not.
        """

        return self._state.pop(key, None) is not None

    def get_model_device(self, model):
        # TODO: move this method to utils.
        # Assumes all parameters are in the same device.
        return next(model.parameters()).device

    def set_optimizer(self, optimizer_conf, **kwargs):
        if bool(kwargs) and type(optimizer_conf) != dict:
            optimizer_conf = dict(
                name=optimizer_conf,
                params=kwargs,
            )
        assert type(optimizer_conf) == dict
        # Save optimizer configuration so we can used later for creating and
        # object or using it as a historical log.
        self._settings[OPTIMIZER] = optimizer_conf
        # Reset optimizer of the state so we need to recreate it before fit.
        self._remove_from_state(OPTIMIZER)
        # Reset schedulers because they depend on the optimizer.
        self._remove_from_state(SCHEDULER)

    def _create_optimizer(self):
        print("Creating optimizer...")
        optimizer_conf = self._settings[OPTIMIZER]
        name_or_optimizer = optimizer_conf["name"]
        params = optimizer_conf.get("params", {})
        optimizer = create_optim(name_or_optimizer,
                                 self.model.parameters(),
                                 **params)
        self._state[OPTIMIZER] = optimizer

    def set_logger(self, logger_conf, **kwargs):
        if bool(kwargs) and type(logger_conf) != dict:
            logger_conf = dict(
                name=logger_conf,
                params=kwargs,
            )
        assert type(logger_conf) == dict
        # Save logger configuration so we can used later for creating and
        # object or using it as a historical log.
        self._settings[TRAIN_LOGGER] = logger_conf
        # Removing logger from the state so we need to recreate it before fit.
        self._remove_from_state(TRAIN_LOGGER)

    def _create_train_logger(self, total_samples):
        print("Creating train logger...")
        logger_conf = self._settings["train_logger"]
        name_or_logger = logger_conf["name"]
        params = logger_conf.get("params", {})
        train_logger = create_logger(name_or_logger,
                                     total_samples,
                                     **params)
        self._state[TRAIN_LOGGER] = train_logger

    def _create_dl_train(self, X, Y, batch_size):
        print("Creating data loader for train...")
        dl_train = get_tensor_data_loader(X, Y,
                                          batch_size=batch_size,
                                          shuffle=True)
        self._state[DL_TRAIN] = dl_train

    def set_lr(self, lr):
        """Utility function to change the learning rate of an optimizer. """

        for i in self._state[OPTIMIZER].param_groups:
            i["lr"] = lr

    def prepare_for_fit(self,
                        data_train_X, data_train_Y=None, batch_size=32):
        """Creates all the objects needed for training using the settings. """

        # Create data loaders if needed.
        if DL_TRAIN not in self._state:
            self._create_dl_train(data_train_X, data_train_Y,
                                  batch_size=batch_size)
        dl_train = self._state[DL_TRAIN]
        # Create train and test loggers.
        if TRAIN_LOGGER not in self._state:
            self._create_train_logger(len(dl_train))
        # Create optimizer if needed.
        if OPTIMIZER not in self._state:
            self._create_optimizer()
        optimizer = self._state[OPTIMIZER]
        # TODO: Create LR schedulers after creating optimizer.
        #if scheduler is not None:
        #    patience = 10 if scheduler is True else scheduler
        #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        # TODO: Get loss function from a factory method.
        loss_function = torch.nn.functional.l1_loss
        self._state[LOSS] = loss_function
        # TODO: Get metric function from a factory method.
        metric_function = sklearn.metrics.mean_absolute_error
        self._state[METRIC] = metric_function

    def fit(self,
            data_train_X, data_train_Y=None,
            data_val_X=None, data_val_Y=None,
            n_epochs=2,
            batch_size=32):
        """Train a PyTorch complex model using a SKLearn simple API. """

        self.prepare_for_fit(data_train_X, data_train_Y, batch_size)
        # Set the fit fundamental tools as local variables.
        optimizer = self._state[OPTIMIZER]
        train_logger = self._state[TRAIN_LOGGER]
        dl_train = self._state[DL_TRAIN]
        loss_function = self._state[LOSS]
        metric_function = self._state[METRIC]

        ###############
        # Train loop. #
        ###############
        device = self.get_model_device(self.model)
        self.model.train()
        for n_epoch in range(1, n_epochs + 1):
            train_logger.init_epoch()
            for X, Y in dl_train:
                train_logger.init_batch()
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                Y_hat = self.model(X)
                loss = loss_function(Y_hat, Y)
                # TODO: apply weights by sample or by class.
                #if len(selected_type) > 1:
                #   loss = loss * batch.w.view(-1)
                #loss = loss.mean()
                loss.backward()
                optimizer.step()
                Y, Y_hat = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()
 
                # Undo transformations before computing metric.
                #y, y_hat = pd.DataFrame({
                #    "molecule_name": np.nan,
                #    "scalar_coupling_constant": y.ravel(),
                #    "type": batch.type[mask].cpu().numpy().ravel()
                #}), pd.DataFrame({
                #    "molecule_name": np.nan,
                #    "scalar_coupling_constant": y_hat.ravel(),
                #    "type": batch.type[mask].cpu().numpy().ravel()
                #})
                #y, y_hat = pipe_y.inverse_transform(y), pipe_y.inverse_transform(y_hat)
                #y, y_hat = y["scalar_coupling_constant"], y_hat["scalar_coupling_constant"]

                metric = metric_function(Y, Y_hat)
                train_logger.log("Metric", metric)
                train_logger.log("Loss", loss.item())
                batch_size = len(Y_hat)
                train_logger.end_batch(batch_size)
            train_logger.end_epoch()
            #if scheduler is not None:
            #    scheduler.step(train_logger.get_current_loss("j-coupling"))
            #if data_loader_val is not None:
            #    print("Validation:")
            #    #self.test(data_loader_val, logger=test_logger)
            #    test_model_batch(ds_val, plot=False)
            #    self.train()
        train_logger.end()

    def test(self, data_loader,
             logger=None, log_epoch_interval=1, log_batch_interval=1e100):
        if logger == None:
            logger = IntervalLogger(epoch_interval=log_epoch_interval,
                                    batch_interval=log_batch_interval)
            logger.init(["j-coupling"], len(data_loader))
        self.eval()
        with torch.no_grad():
            device = self.get_device()
            logger.init_epoch()
            for batch in data_loader:
                logger.init_batch()
                batch = batch.to(device)
                mask = batch.y_mask
                if self.selected_type is not None:
                    mask_type = torch.zeros(len(mask), dtype=torch.uint8).to(device)
                    for i in self.selected_type:
                        mask_type |= (batch.type == i).view(-1)
                    mask *= mask_type
                y_hat = self(batch.x, batch.edge_index, batch.edge_attr, batch.type, mask).view(-1)
                y = batch.y[mask]
                loss = torch.nn.functional.l1_loss(y_hat, y, reduction="none")
                if len(selected_type) > 1:
                    loss = loss * batch.w[mask]
                #loss = torch.nn.functional.mse_loss(y_hat, y)
                loss = loss.mean()
                y, y_hat = y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
                
                y, y_hat = pd.DataFrame({
                    "molecule_name": np.nan,
                    "scalar_coupling_constant": y.ravel(),
                    "type": batch.type[mask].cpu().numpy().ravel()
                }), pd.DataFrame({
                    "molecule_name": np.nan,
                    "scalar_coupling_constant": y_hat.ravel(),
                    "type": batch.type[mask].cpu().numpy().ravel()
                })
                y, y_hat = pipe_y.inverse_transform(y), pipe_y.inverse_transform(y_hat)
                y, y_hat = y["scalar_coupling_constant"], y_hat["scalar_coupling_constant"]
                
                metrics = sklearn.metrics.mean_absolute_error(y, y_hat)
                batch_size = len(y_hat)
                logger.end_batch(batch_size, [loss.item()], [metrics])
            logger.end_epoch()
            logger.end()
        self.test_logger = logger

    def predict(self, data):
        # TODO: dynamic batch size and iterate batch and join solutions.
        dl = get_tensor_data_loader(data, batch_size=1000000)
        self.model.eval()
        with torch.no_grad():
            device = self.get_model_device(self.model)
            for X, in dl:
                X = X.to(device)
                Y_hat = self.model(X)
                return Y_hat


def test_model(data_set, plot=False, rescale=True, plot_y=True):
    y, y_hat, mask, t = model.predict(data_set)
    y, y_hat, t = y[mask], y_hat[mask], t[mask]
    y, y_hat = pd.DataFrame({
        "molecule_name": np.nan,
        "scalar_coupling_constant": y.cpu().numpy().squeeze(),
        "type": t.cpu().numpy().squeeze()
    }), pd.DataFrame({
        "molecule_name": np.nan,
        "scalar_coupling_constant": y_hat.cpu().numpy().squeeze(),
        "type": t.cpu().numpy().squeeze()
    })
    if rescale:
        y, y_hat = pipe_y.inverse_transform(y), pipe_y.inverse_transform(y_hat)
    df_results = pd.DataFrame({"y": y["scalar_coupling_constant"], "y_hat": y_hat["scalar_coupling_constant"], "type": y["type"]})
    # Plot histogram
    if plot:
        if plot_y:
            y["scalar_coupling_constant"].hist(bins=50)
        ax = y_hat["scalar_coupling_constant"].hist(bins=50)
        _ = ax.set_xticklabels(ax.get_xticks(), rotation=45)
    # Show metric
    print("# rows:", y.shape[0])
    print(compute_metrics(y["scalar_coupling_constant"], y_hat["scalar_coupling_constant"], y["type"]))

    optimizer = nker(seq, size)
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def test_model_batch(data_set, plot=False, rescale=True, plot_y=True, plot_y_hat=True):
    y, y_hat, mask, t = [],[],[],[]
    for i in chunker(data_set, 1024):
        y_, y_hat_, mask_, t_ = model.predict(i)
        y.append(y_)
        y_hat.append(y_hat_)
        mask.append(mask_)
        t.append(t_)
    y, y_hat, mask, t = torch.cat(y), torch.cat(y_hat), torch.cat(mask), torch.cat(t)
    y, y_hat, t = y[mask], y_hat[mask], t[mask]
    y, y_hat = pd.DataFrame({
        "molecule_name": np.nan,
        "scalar_coupling_constant": y.cpu().numpy().squeeze(),
        "type": t.cpu().numpy().squeeze()
    }), pd.DataFrame({
        "molecule_name": np.nan,
        "scalar_coupling_constant": y_hat.cpu().numpy().squeeze(),
        "type": t.cpu().numpy().squeeze()
    })
    if rescale:
        y, y_hat = pipe_y.inverse_transform(y), pipe_y.inverse_transform(y_hat)
    df_results = pd.DataFrame({"y": y["scalar_coupling_constant"], "y_hat": y_hat["scalar_coupling_constant"], "type": y["type"]})
    # Plot histogram
    if plot:
        if plot_y:
            y["scalar_coupling_constant"].hist(bins=50)
        if plot_y_hat:
            ax = y_hat["scalar_coupling_constant"].hist(bins=50)
            _ = ax.set_xticklabels(ax.get_xticks(), rotation=45)
    # Show metric
    print("# rows:", y.shape[0])
    print(compute_metrics(y["scalar_coupling_constant"], y_hat["scalar_coupling_constant"], y["type"]))    

def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):
    """Compute the Kaggle metric. """
    maes = (y_true - y_pred).abs().groupby(groups).mean()
    print(maes)
    return np.log(maes.map(lambda x: max(x, floor))).mean()

def compute_metrics(y_true, y_pred, groups):
    return pd.Series({
        "group_mean_log_mae": group_mean_log_mae(y_true, y_pred, groups),
        "mae": sklearn.metrics.mean_absolute_error(y_true, y_pred),
        "mse": sklearn.metrics.mean_squared_error(y_true, y_pred),
    })

def trunc_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

