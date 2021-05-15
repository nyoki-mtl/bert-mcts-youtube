from .mlm import MLMModule, MLMDataModule
from .policy_value import PolicyValueModule, PolicyValueDataModule


def get_pl_modules(cfg):
    if cfg.model_type == 'MLM':
        return MLMModule(cfg), MLMDataModule(cfg)
    elif cfg.model_type == 'PolicyValue':
        return PolicyValueModule(cfg), PolicyValueDataModule(cfg)
    else:
        raise NotImplementedError
