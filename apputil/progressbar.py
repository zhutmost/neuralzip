from time import time

import pytorch_lightning as pl


def _serialize_metrics(progressbar_log_dict, filter_fn=None):
    if filter_fn:
        progressbar_log_dict = {k: v for k, v in progressbar_log_dict.items() if filter_fn(k)}
    msg = ''
    for metric, value in progressbar_log_dict.items():
        if type(value) is str:
            msg += f'{metric}: {value}  '
        elif 'acc' in metric:
            msg += f'{metric}: {value:.3%}  '
        else:
            msg += f'{metric}: {value:f}  '
    return msg


class ProgressBar(pl.callbacks.ProgressBarBase):
    def __init__(self, logger, refresh_rate: int = 50):
        super().__init__()
        self._logger = logger
        self._refresh_rate = refresh_rate
        self._enabled = True

        # a time flag to indicate the beginning of an epoch
        self._time = 0

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        # No need to disable the ProgressBar on processes with LOCAL_RANK != 1, because the
        # StreamHandler of logging is disabled on these processes.
        self._enabled = True

    def enable(self) -> None:
        self._enabled = True

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self._logger.info(f'Trainer fit begins ... '
                          f'Current epoch: {trainer.current_epoch}, batch: {self.train_batch_idx}')

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf') and not trainer.fast_dev_run:
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        self._logger.info(f'\n                   '
                          f'>>> >>> >>> >>> Epoch {trainer.current_epoch}, including {total_batches} batches '
                          f'(train: {total_train_batches} & val: {total_val_batches}) '
                          f'<<< <<< <<< <<<')
        self._time = time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            batch_time = (time() - self._time) / self.train_batch_idx
            msg = f'Train (Epoch {trainer.current_epoch}, ' \
                  f'Batch {self.train_batch_idx} / {self.total_train_batches}, {batch_time:.2f}s/it) => '
            msg += _serialize_metrics(trainer.progress_bar_dict,
                                      filter_fn=lambda x: not x.startswith('val_') and not x.startswith('test_'))
            self._logger.info(msg)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._logger.info(f'Trainer fit ends.')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            self._logger.info(f'\n                   '
                              f'>>> Validate step begins ... Epoch {trainer.current_epoch}, '
                              f'including {self.total_val_batches} batches')
        self._time = time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.val_batch_idx % self.refresh_rate == 0:
            batch_time = (time() - self._time) / self.val_batch_idx
            msg = f'Validate (Epoch {trainer.current_epoch}, ' \
                  f'Batch {self.val_batch_idx} / {self.total_val_batches}, {batch_time:.2f}s/it) => '
            msg += _serialize_metrics(trainer.progress_bar_dict,
                                      filter_fn=lambda x: x.startswith('val_') and x.endswith('_step'))
            self._logger.info(msg)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if not trainer.running_sanity_check:
            msg = '>>> Validate ends => '
            msg += _serialize_metrics(trainer.progress_bar_dict,
                                      filter_fn=lambda x: x.startswith('val_') and x.endswith('_epoch'))
            self._logger.info(msg)

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self._logger.info(f'\n                   >>> >>> >>> >>> Test, '
                          f'including {self.total_test_batches} batches '
                          f'<<< <<< <<< <<<')
        self._time = time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.test_batch_idx % self.refresh_rate == 0:
            batch_time = (time() - self._time) / self.test_batch_idx
            msg = f'Test (Batch {self.test_batch_idx} / {self.total_test_batches}, {batch_time:.2f}s/it) => '
            msg += _serialize_metrics(trainer.progress_bar_dict,
                                      filter_fn=lambda x: x.startswith('test_') and x.endswith('_step'))
            self._logger.info(msg)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        msg = '>>> Test ends => '
        msg += _serialize_metrics(trainer.progress_bar_dict,
                                  filter_fn=lambda x: x.startswith('test_') and x.endswith('_epoch'))
        self._logger.info(msg + '\n')

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self._logger.info('Validate set sanity check begins.')

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self._logger.info('Validate set sanity check ends.')
