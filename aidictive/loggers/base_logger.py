
import time


class BaseLogger(object):

    def __init__(self):
        pass

    def init(self, total_samples=None):
        self.total_samples = total_samples
        # Properties related to the epoch.
        self.epoch = 0
        self.epoch_init_time = None
        self.epoch_end_time = None
        self.epoch_samples = 0
        self.epoch_metrics = {}
        self.epoch_metrics_hist = []
        # Properties rated to the batch.
        self.batch = 0
        self.batch_init_time = None
        self.batch_end_time = None

    def end(self):
        pass

    def init_epoch(self):
        self.epoch_init_time = time.time()
        self.epoch += 1
        self.epoch_samples = 0
        # Reset batch counter.
        self.batch = 0
        # Reset metric dict.
        self.epoch_metrics = {}

    def init_batch(self):
        self.batch_init_time = time.time()
        self.batch += 1
        self.batch_metrics = {}

    def log(self, metric, value):
        self.batch_metrics[metric] = value

    def end_batch(self, samples):
        self.batch_end_time = time.time() - self.batch_init_time
        self.epoch_samples += samples
        # Add batch metrics to the epoch metrics.
        for k in self.batch_metrics:
            value = self.batch_metrics[k]
            last_value = self.epoch_metrics.get(k, 0)
            self.epoch_metrics[k] = value * samples + last_value

    def end_epoch(self):
        self.end_epoch_time = time.time() - self.epoch_init_time
        # Normalize all total metrics by number of samples.
        self.epoch_metrics = {
                k: self.get_metric_by_sample(k) for k in self.epoch_metrics}
        # Add extra information to the metric dict before saving it.
        self.epoch_metrics["epoch"] = self.epoch
        # Keep metrics dict as historic data.
        self.epoch_metrics_hist.append(self.epoch_metrics)

    def get_metric_by_sample(self, target_name):
        return self.epoch_metrics[target_name] / self.epoch_samples

