
import operator
import warnings

import numpy as np

import sklearn
import sklearn.metrics

import torch

from aidictive import utils
from aidictive.optimizers import create as create_optim
from aidictive.loggers import create as create_logger
from aidictive.data import get_tensor_data_loader
from aidictive.schedulers import create as create_scheduler
from aidictive.losses import get as create_loss
from aidictive.metrics import get as create_metric


# Keys used to store settings and object on settings and state dict.
OPTIMIZER = "optimizer"
SCHEDULER = "scheduler"
TRAIN_LOGGER = "train_logger"
TEST_LOGGER = "test_logger"
DL_TRAIN = "dl_train"
DL_TEST = "dl_test"
LOSS = "loss"
METRIC = "metric"

# Default configurations.
DEFAULT_OPTIMIZER = dict(
    name="radam"
)
DEFAULT_TRAIN_LOGGER = dict(
    name="interval",
    params=dict(
        epoch_interval=1,
        batch_interval=1e100,
    )
)
DEFAULT_TEST_LOGGER = dict(
    name="interval",
    params=dict(
        epoch_interval=1,
        batch_interval=1e100,
    )
)
DEFAULT_SCHEDULER = dict(
    name="reducelronplateau",
    params=dict(
        patience=5,
        factor=0.1,
        verbose=True,
    )
)
DEFAULT_DL_PARAMS = {
    DL_TRAIN: dict(
        batch_size=32,
        shuffle=True,
    ),
    DL_TEST: dict(
        batch_size=128,
    )
}


class Trainer(object):

    def __init__(self, model, gpu=True):
        # Move model to GPU if GPU is present.
        if gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
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
        self.set_scheduler(None)
        self.set_train_logger(DEFAULT_TRAIN_LOGGER)
        self.set_test_logger(DEFAULT_TEST_LOGGER)

    def _remove_from_state(self, key):
        """Safety removes a key from the state dict.

        Returns True if the key was in the state, False if not.
        """

        return self._remove_from_dict(self._state, key)

    def _remove_from_settings(self, key):
        """Safety removes a key from the settings dict.

        Returns True if the key was in the settings, False if not.
        """

        return self._remove_from_dict(self._settings, key)

    def _remove_from_dict(self, dic, key):
        """Safety removes a key from a dictionary.
        
        Returns True if the key was in the dict, False if not.
        """

        return dic.pop(key, None) is not None

    def set_optimizer(self, optimizer_conf, **kwargs):
        if type(optimizer_conf) != dict:
            optimizer_conf = dict(
                name=optimizer_conf,
                params=kwargs,
            )
        assert type(optimizer_conf) == dict
        # Save optimizer configuration so we can use it later for creating an
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

    def set_scheduler(self, scheduler_conf, **kwargs):
        if bool(kwargs) and type(scheduler_conf) != dict:
            scheduler_conf = dict(
                name=scheduler_conf,
                params=kwargs,
            )
        if scheduler_conf is None:
            # Remove scheduler if exists.
            self._remove_from_settings(SCHEDULER)
        else:
            assert type(scheduler_conf) == dict
            # Save scheduler configuration so we can use it later for creating
            # an object or using it as a historical log.
            self._settings[SCHEDULER] = scheduler_conf
        # Reset scheduler of the state so we need to recreate it before fit.
        self._remove_from_state(SCHEDULER)

    def _create_scheduler(self, optimizer):
        print("Creating scheduler...")
        scheduler_conf = self._settings[SCHEDULER]
        name_or_scheduler = scheduler_conf["name"]
        params = scheduler_conf.get("params", {})
        scheduler = create_scheduler(name_or_scheduler,
                                     optimizer,
                                     **params)
        self._state[SCHEDULER] = scheduler

    def _set_logger(self, key, logger_conf, **kwargs):
        if bool(kwargs) and type(logger_conf) != dict:
            logger_conf = dict(
                name=logger_conf,
                params=kwargs,
            )
        assert type(logger_conf) == dict
        # Save logger configuration so we can used later for creating and
        # object or using it as a historical log.
        self._settings[key] = logger_conf
        # Removing logger from the state so we need to recreate it before fit.
        self._remove_from_state(key)

    def set_train_logger(self, logger_conf, **kwargs):
        self._set_logger(TRAIN_LOGGER, logger_conf, **kwargs)

    def set_test_logger(self, logger_conf, **kwargs):
        self._set_logger(TEST_LOGGER, logger_conf, **kwargs)

    def _create_logger(self, key_settings, key_state, total_samples):
        print("Creating logger...")
        logger_conf = self._settings[key_settings]
        name_or_logger = logger_conf["name"]
        params = logger_conf.get("params", {})
        logger = create_logger(name_or_logger,
                               total_samples,
                               **params)
        self._state[key_state] = logger

    def _create_train_logger(self, total_samples):
        self._create_logger(TRAIN_LOGGER, TRAIN_LOGGER, total_samples)

    def _create_test_logger(self, total_samples, during_train=False):
        settings = TRAIN_LOGGER if during_train else TEST_LOGGER
        self._create_logger(settings, TEST_LOGGER, total_samples)

    def set_loss(self, loss_conf, **kwargs):
        self._settings[LOSS] = loss_conf
        self._remove_from_state(LOSS)

    def _create_loss(self):
        self._state[LOSS] = create_loss(self._settings[LOSS])

    def set_metric(self, metric_conf, **kwargs):
        self._settings[METRIC] = metric_conf
        self._remove_from_state(METRIC)

    def _create_metric(self):
        self._state[METRIC] = create_metric(self._settings[METRIC])

    def set_lr(self, lr):
        """Utility function to change the learning rate of an optimizer. """

        if OPTIMIZER in self._state:
            for i in self._state[OPTIMIZER].param_groups:
                i["lr"] = lr
        else:
            self._settings[OPTIMIZER]["params"]["lr"] = lr

    def set_train_data(self, *args, **kwargs):
        self._set_data(DL_TRAIN, *args,**kwargs)

    def set_test_data(self, *args, **kwargs):
        self._set_data(DL_TEST, *args, **kwargs)

    def _set_data(self, key, X, Y=None, **dl_params):
        # TODO: bug when X is a pytorch dataset. FIXME!!!!
        only_X = False
        if X is None:
            # Nothing to do here. If the data is required (like training set
            # for fitting) you will get an error later.
            return
        elif type(X) == dict:
            new_s = X
            only_X = True
        elif isinstance(X, torch.utils.data.dataloader.DataLoader):
            new_s = dict(dl=X)
            only_X = True
        else:
            dl_params_default = DEFAULT_DL_PARAMS[key].copy()
            dl_params_default.update(dl_params)
            new_s = dict(X=X, Y=Y, dl_params=dl_params_default)
        # If all your data is in X, check that you are not using any other of
        # the params to avoid errors.
        if only_X and (any(dl_params.values()) or Y is not None):
            warnings.warn("You are already using a data loader as data source,"
                          f" ignoring given data loader params: {dl_params}")
        if key in self._settings:
            old_s = self._settings[key]
            if operator.eq(new_s, old_s):
                # Same training data, not remove from state.
                return
        # Different or new training data, update settings and state.
        self._settings[key] = new_s
        self._remove_from_state(key)

    def _create_dl(self, dl_settings):
        if "dl" in dl_settings:
            return dl_settings["dl"]

        print("Creating data loader...")
        X, Y = dl_settings["X"], dl_settings["Y"]
        dl_params = dl_settings.get("dl_params", {})
        return get_tensor_data_loader(X, Y, **dl_params)

    def _create_dl_train(self):
        dl_train_settings = self._settings[DL_TRAIN]
        self._state[DL_TRAIN] = self._create_dl(dl_train_settings)

    def _create_dl_test(self):
        dl_test_settings = self._settings[DL_TEST]
        self._state[DL_TEST] = self._create_dl(dl_test_settings)

    def prepare_for_fit(self):
        """Creates all the objects needed for training using the settings. """

        # Create data loaders if needed.
        if DL_TRAIN not in self._state:
            if DL_TRAIN not in self._settings:
                raise Exception("I need traning data to learn something!")
            self._create_dl_train()
        dl_train = self._state[DL_TRAIN]
        # Create train and test loggers.
        if TRAIN_LOGGER not in self._state:
            self._create_train_logger(len(dl_train))
        #if TEST_LOGGER not in self._state:
        #    self._create_test_logger(len(dl_val), during_train=True)
        # Create optimizer if needed.
        if OPTIMIZER not in self._state:
            self._create_optimizer()
        optimizer = self._state[OPTIMIZER]
        # Create LR scheduler (always after creating optimizer).
        if SCHEDULER in self._settings and SCHEDULER not in self._state:
            self._create_scheduler(optimizer)
        # Get loss function from a factory method.
        if LOSS not in self._state:
            self._create_loss()
        # Get metric function from a factory method.
        if METRIC not in self._state:
            self._create_metric()

    def fit(self,
            data_train_X=None, data_train_Y=None, batch_size=None,
            data_val_X=None, data_val_Y=None, batch_size_val=None,
            n_epochs=1):
        """Train a PyTorch complex model using a SKLearn simple API. """

        # Set a new training and test data if needed.
        self.set_train_data(data_train_X, data_train_Y, batch_size=batch_size)
        self.set_test_data(data_val_X, data_val_Y, batch_size=batch_size_val)
        # Using the settings, create all the objects needed for training.
        self.prepare_for_fit()
        # Set the fit fundamental tools as local variables.
        optimizer = self._state[OPTIMIZER]
        scheduler = self._state.get(SCHEDULER, None)
        train_logger = self._state[TRAIN_LOGGER]
        dl_train = self._state[DL_TRAIN]
        loss_function = self._state[LOSS]
        metric_function = self._state[METRIC]

        ###############
        # Train loop. #
        ###############
        device = utils.get_model_device(self.model)
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

                if len(Y_hat.shape) > len(Y.shape):
                    Y_hat = np.argmax(Y_hat, axis=1)
                metric = metric_function(Y, Y_hat)
                train_logger.log("metric", metric)
                train_logger.log("loss", loss.item())
                batch_size = len(Y_hat)
                train_logger.end_batch(batch_size)
            train_logger.end_epoch()
            if scheduler is not None:
                scheduler.step(train_logger.get_metric_by_sample("loss"))
            # TODO: validation error measure. Create a new method using the
            # same logger.
            #if data_loader_val is not None:
            #    print("Validation:")
            #    #self.test(data_loader_val, logger=test_logger)
            #    test_model_batch(ds_val, plot=False)
            #    self.train()
        train_logger.end()

    def prepare_for_test(self):
        # Create data loaders if needed.
        if DL_TEST not in self._state:
            if DL_TEST not in self._settings:
                raise Exception("I need test data to test something!")
            self._create_dl_test()
        dl_test = self._state[DL_TEST]
        # Create test logger.
        if TEST_LOGGER not in self._state:
            self._create_test_logger(len(dl_test))
        # Get loss function from a factory method.
        if LOSS not in self._state:
            self._create_loss()
        # Get metric function from a factory method.
        if METRIC not in self._state:
            self._create_metric()

    def test(self, data_test_X=None, data_test_Y=None, batch_size=None):
        """Test your model easily with this method. """

        # Set test data if needed.
        self.set_test_data(data_test_X, data_test_Y, batch_size=batch_size)
        # Create all objects needed for test.
        self.prepare_for_test()
        # Set the test fundamental tools as local variables.
        test_logger = self._state[TEST_LOGGER]
        dl_test = self._state[DL_TEST]
        loss_function = self._state[LOSS]
        metric_function = self._state[METRIC]

        self._test(dl_test, loss_function, metric_function, test_logger)

    def test_train(self):
        test_logger = self._state[TEST_LOGGER]
        dl_train = self._state[DL_TRAIN]
        loss_function = self._state[LOSS]
        metric_function = self._state[METRIC]
        self._test(dl_train, loss_function, metric_function, test_logger)

    def _test(self, dl, loss_function, metric_function, logger):
        device = utils.get_model_device(self.model)
        self.model.eval()
        with torch.no_grad():
            logger.init_epoch()
            for X, Y in dl:
                logger.init_batch()
                # Forward pass.
                X, Y = X.to(device), Y.to(device)
                Y_hat = self.model(X)
                loss = loss_function(Y_hat, Y)
                # Compute metric.
                Y = Y.detach().cpu().numpy()
                Y_hat = Y_hat.detach().cpu().numpy()
                if len(Y_hat.shape) > len(Y.shape):
                    Y_hat = np.argmax(Y_hat, axis=1)
                metric = metric_function(Y, Y_hat)
                # Log results.
                logger.log("metric", metric)
                logger.log("loss", loss.item())
                batch_size = len(Y_hat)
                logger.end_batch(batch_size)
            logger.end_epoch()
            logger.end()

    def predict(self, data):
        # TODO: dynamic batch size and iterate batch and join solutions.
        dl = get_tensor_data_loader(data, batch_size=1000000)
        self.model.eval()
        with torch.no_grad():
            device = utils.get_model_device(self.model)
            for X, in dl:
                X = X.to(device)
                Y_hat = self.model(X)
                return Y_hat

    def predict_argmax(self, preds):
        return torch.argmax(preds, axis=1)

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

