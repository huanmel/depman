"""Registry of available config backends."""
from typing import Optional

from depman.backends.base import ConfigBackend
from depman.backends.gitman_backend import GitmanBackend

BACKENDS = [GitmanBackend()]


def get_backend(name: str) -> Optional[ConfigBackend]:
    for backend in BACKENDS:
        if backend.name == name:
            return backend
    return None
