from logging import Logger
from time import time
from typing import Callable, Optional

import pytorch_lightning as pl


class ProgressBar(pl.callbacks.ProgressBarBase):
    """A custom ProgressBar to log the training progress."""
    def __init__(self, logger: Logger, refresh_rate: int = 50) -> None:
        """Create a ProgressBar.

        The ProgressBar provided by Lightning is based on tqdm. Its output always roll over the
        previous information, and the printed logs are too brief. This custom one serializes all the
        metrics provided by user, and the outputs are much more detailed. The logs are delivered to
        a Logging.logger (rather than printed to CLI directly), which can easily captured into a log
        file.

        Args:
            logger: A logging.Logger to record the training log.
            refresh_rate: Determines at which rate (in number of batches) the progress bars get
                updated. Set it to ``0`` to disable the display.
        """
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

    @staticmethod
    def _serialize_metrics(progressbar_log_dict: dict, filter_fn: Optional[Callable[[str], bool]] = None) -> str:
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

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self._logger.info('Validate set sanity check begins.')

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self._logger.info('Validate set sanity check ends.')

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self._logger.info(f'Trainer fit begins ... '
                          f'Current epoch: {trainer.current_epoch}, batch: {self.train_batch_idx}')

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self._logger.info(f'\n\n                   '
                          f'>>> >>> >>> >>> Epoch {trainer.current_epoch}, '
                          f'including {self.total_batches_current_epoch} batches '
                          f'(train: {self.total_train_batches}, val: {self.total_val_batches}) '
                          f'<<< <<< <<< <<<')
        self._time = time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        current = self.train_batch_idx + self._val_processed
        if self._should_update(current, self.total_batches_current_epoch):
            batch_time = (time() - self._time) / self.train_batch_idx
            msg = f'{self.train_description} (Epoch {trainer.current_epoch}, ' \
                  f'Batch {self.train_batch_idx} / {self.total_train_batches}, {batch_time:.2f}s/it) => '
            msg += self._serialize_metrics(trainer.progress_bar_metrics,
                                           filter_fn=lambda x: not x.startswith('val_') and not x.startswith('test_'))
            self._logger.info(msg)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._logger.info(f'Trainer fit ends.')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.sanity_checking:
            self._logger.info(f'\n                   '
                              f'>>> {self.validation_description} step begins ... Epoch {trainer.current_epoch}, '
                              f'including {self.total_val_batches} batches')
        self._time = time()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.has_dataloader_changed(dataloader_idx):
            return
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self._logger.info(f"Current {desc} dataloader index: {self._current_eval_dataloader_idx}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.val_batch_idx, self.total_val_batches_current_dataloader):
            batch_time = (time() - self._time) / self.val_batch_idx
            msg = f'{self.validation_description} (Epoch {trainer.current_epoch}, ' \
                  f'Batch {self.val_batch_idx} / {self.total_val_batches}, {batch_time:.2f}s/it) => '
            msg += self._serialize_metrics(trainer.progress_bar_metrics,
                                           filter_fn=lambda x: x.startswith('val_') and x.endswith('_step'))
            self._logger.info(msg)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if not trainer.sanity_checking:
            msg = '>>> Validate ends => '
            msg += self._serialize_metrics(trainer.progress_bar_metrics,
                                           filter_fn=lambda x: x.startswith('val_') and x.endswith('_epoch'))
            self._logger.info(msg)
        self.reset_dataloader_idx_tracker()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self._logger.info(f'\n\n                   '
                          f'>>> >>> >>> >>> {self.test_description} '
                          f'<<< <<< <<< <<<')
        self._time = time()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if not self.has_dataloader_changed(dataloader_idx):
            return
        self._logger.info(f"Current {self.test_description} dataloader index: {self._current_eval_dataloader_idx}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.test_batch_idx, self.total_test_batches_current_dataloader):
            batch_time = (time() - self._time) / self.test_batch_idx
            msg = f'{self.test_description} (Batch {self.test_batch_idx} / {self.total_test_batches_current_dataloader}, {batch_time:.2f}s/it) => '
            msg += self._serialize_metrics(trainer.progress_bar_metrics,
                                           filter_fn=lambda x: x.startswith('test_') and x.endswith('_step'))
            self._logger.info(msg)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        msg = f'>>> {self.test_description} ends => '
        msg += self._serialize_metrics(trainer.progress_bar_metrics,
                                       filter_fn=lambda x: x.startswith('test_') and x.endswith('_epoch'))
        self._logger.info(msg + '\n')
        self.reset_dataloader_idx_tracker()

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # don't show the version number
        items.pop("v_num", None)
        return items

    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def print(self, *args, sep: str = " ", **kwargs):
        s = sep.join(map(str, args))
        self._logger.info(f"[Progress Print] {s}")
