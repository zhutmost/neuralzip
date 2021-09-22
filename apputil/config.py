import sys
from pathlib import Path
from typing import Optional, Union

from omegaconf import OmegaConf, DictConfig


def get_config(base_conf_filepath: Optional[Union[Path, str]] = None) -> DictConfig:
    """Get user configurations from CLI parameters and overwrite the default ones.

    Args:
        base_conf_filepath: The path to the template configuration file, which provides default
            values of all the configuration items.

    Returns:
        A DictConfig object, which should include all the settings needed by NeuralZip.
    """
    conf = OmegaConf.load(base_conf_filepath) if base_conf_filepath else OmegaConf.create()
    conf_files = [i[len('conf_filepath='):] for i in sys.argv if i.startswith('conf_filepath=')]
    for conf_file in conf_files:
        cfg_load = OmegaConf.load(conf_file)
        conf.merge_with(cfg_load)

    conf_cli_override = [i for i in sys.argv[1:] if not i.startswith('conf_filepath=')]
    conf.merge_with_dotlist(conf_cli_override)

    return conf
