import logging

import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_runtime():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
    )
    yield
    ray.shutdown()
