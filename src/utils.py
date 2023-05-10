import logging

LOGGER = logging.getLogger(__name__)


def foo():
    LOGGER.info("Foo")
    return True
