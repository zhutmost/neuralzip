from omegaconf import OmegaConf
import sys


def get_config(base_conf_filepath=None):
    conf = OmegaConf.load(base_conf_filepath) if base_conf_filepath else OmegaConf.create()
    conf_files = [i[len('conf_filepath='):] for i in sys.argv if i.startswith('conf_filepath=')]
    for conf_file in conf_files:
        cfg_load = OmegaConf.load(conf_file)
        conf.merge_with(cfg_load)

    conf_cli_override = [i for i in sys.argv[1:] if not i.startswith('conf_filepath=')]
    conf.merge_with_dotlist(conf_cli_override)

    return conf
