import logging


def enable_verbose_logging(level: int = logging.INFO) -> None:
    """
    Attach a StreamHandler to the `surrogate_index` logger
    if one doesn't already exist. Used for verbose=True mode.
    """
    pkg_logger = logging.getLogger("surrogate_index")

    if pkg_logger.handlers:
        pkg_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False  # don't send to root logger (avoid double prints)
