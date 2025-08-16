# logging.py
import logging
import logging.config
from os import makedirs
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose


def setup_logging():
    """Setup basic logging configuration with file and console outputs."""
    # Create directory for logs if it doesn't exist
    with initialize(config_path="../config/logs", version_base=None):
        cfg = compose(config_name="default")

    makedirs("../logs", exist_ok=True)
    log_cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.config.dictConfig(log_cfg)

    # Get the root logger
    logger = logging.getLogger(__name__)
    logger.info("Logger is up and running.")
