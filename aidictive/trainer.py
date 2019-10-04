
import torch


class Trainer(object):

    def __init__(self):
        self.train_logger = None
        self.test_logger = None
        self.optimizer = None

    def get_optimizer(self, optimizer="sgd", lr=0.01, momentum=0, l2=0):
        """Creates a new optimizer or reuses the given object. """

        if type(optimizer) == str:
            optimizer = optimizer.lower()
            if optimizer == "sgd":
                optimizer = torch.optim.SGD(self.parameters(),
                                            lr=lr,
                                            momentum=momentum,
                                            weight_decay=l2)
            elif optimizer == "adam":
                optimizer = torch.optim.Adam(self.parameters(),
                                             lr=lr,
                                             weight_decay=l2)
            elif optimizer == "radam":
                optimizer = RAdam(self.parameters(),
                                  lr=lr,
                                  weight_decay=l2)
        # If no optimizer is given try to reuse an optimizer from the last
        # executions.
        if optimizer is None:
            optimizer = self.optimizer
        return optimizer

    def get_data_loader(self, data):
        if type(data) != torch.utils.data.DataLoader:
            raise new Exception("Unknown data, I cannot convert type "
                                f"{type(data)} to torch.utils.data.DataLoader")
        return data

    def fit(self, data_train, data_val=None,
            lr=0.01, momentum=0, l2=0, optimizer=None, scheduler=None,
            n_epochs=2,
            logger=None, log_epoch_interval=1, log_batch_interval=1e100, skip_step=False):
        # Set optimizer.
        optimizer = self.get_optimizer(optimizer, lr, momentum, l2)
        # Create train and test loggers.
        if logger == None:
            logger = IntervalLogger(epoch_interval=log_epoch_interval,
                                    batch_interval=log_batch_interval)
            logger.init(["j-coupling"], len(data_loader))
        self.train_logger = logger
        test_logger = None
        if data_loader_val is not None:
            test_logger = IntervalLogger(epoch_interval=log_epoch_interval,
                                         batch_interval=1e100)
            test_logger.init(["j-coupling"], len(data_loader_val))
        self.test_logger = test_logger
        # LR schedulers.
        if scheduler is not None:
            patience = 10 if scheduler is True else scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        # Train loop.
        device = self.get_device()
        self.train()
        for n_epoch in range(1, n_epochs + 1):
            logger.init_epoch()
            for batch in data_loader:
                logger.init_batch()
                batch = batch.to(device)
                optimizer.zero_grad()
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
                    loss = loss * batch.w[mask].view(-1)
                #loss = torch.nn.functional.mse_loss(y_hat, y)
                loss = loss.mean()
                loss.backward()
                if skip_step is False: optimizer.step()
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
            if scheduler is not None:
                scheduler.step(logger.get_current_loss("j-coupling"))
            if data_loader_val is not None:
                print("Validation:")
                #self.test(data_loader_val, logger=test_logger)
                test_model_batch(ds_val, plot=False)
                self.train()
        logger.end()

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

    def predict(self, data_set):
        dl = tg.data.DataLoader(data_set, 1000000)
        self.eval()
        with torch.no_grad():
            device = self.get_device()
            for batch in dl:
                batch = batch.to(device)
                mask = batch.y_mask
                if self.selected_type is not None:
                    mask_type = torch.zeros(len(mask), dtype=torch.uint8).to(device)
                    for i in self.selected_type:
                        mask_type |= (batch.type == i).view(-1)
                    mask *= mask_type
                y_hat = self(batch.x, batch.edge_index, batch.edge_attr, batch.type).view(-1)
                return batch.y, y_hat, mask, batch.type


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

optimizer = nker(seq, size):
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

