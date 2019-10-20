
from .base_logger import BaseLogger


class IntervalLogger(BaseLogger):

    def __init__(self, epoch_interval=1, batch_interval=1e100):
        super().__init__()
        self.epoch_interval = epoch_interval
        # If epoch interval is not 1 we don't want to log any epoch.
        # I suppose that no one is going to run 1e100 epochs...
        self.batch_interval = batch_interval if epoch_interval == 1 else 1e100

    def end_epoch(self):
        super().end_epoch()
        if self.log_epoch():
            print(f"Epoch {self.epoch} ended in {self.end_epoch_time:0.4f}s."
                  f" Batches: {self.batch}.")
            self.log_metrics()

    def end_batch(self, samples):
        super().end_batch(samples)
        if self.log_batch():
            print(f"Batch {self.batch} ended in {self.end_batch_time:0.4f}s.")
            self.log_metrics()

    def log_metrics(self):
        for metric, value in self.epoch_metrics.items():
            print(f"{metric}:\t{value}")

    def log_epoch(self):
        return self.epoch % self.epoch_interval == 0

    def log_batch(self):
        return self.batch % self.batch_interval == 0

