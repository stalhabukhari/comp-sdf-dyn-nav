import json
import time
import functools
import pickle
from enum import Enum
from pathlib import Path
from typing import Dict, Union, TypeVar

import numpy as np
import yaml
from pydantic import BaseModel
from rich import print, console

BaseModelDerivative = TypeVar("BaseModelDerivative", bound="BaseModel")
STOP_COND_DIST = 0.1  # meters
STOP_COND_TIME = 1800  # seconds
STOP_COND_COLL = 300  # collisions
COUNTER = 0  # count sim steps
COLLISIONS = 0  # count collisions

# for dynamic obstacles sim
OBS_DISP_COUNT = 0
DYNAMIC_OBS = False


class TransformFormats(str, Enum):
    OpenGL = "open_gl"
    iGibson = "igibson"


def parse_yaml_file(filepath: Path) -> Dict:
    assert filepath.exists(), f"Non-existent yaml file: {filepath}"
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_string_to_file(string: str, filepath: Path):
    with open(str(filepath), "a") as file:
        file.write(string + "\n")


def save_numpy_array_to_disk(array: np.ndarray, filepath: Union[str, Path]):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert filepath.suffix == ".npy", f"Invalid numpy file extension: {filepath.suffix}"
    np.save(str(filepath), array)


def load_numpy_array_from_disk(filepath: Union[str, Path]) -> np.ndarray:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert filepath.suffix == ".npy", f"Invalid numpy file extension: {filepath.suffix}"
    return np.load(str(filepath))


def load_pickle_from_disk(filepath: Union[str, Path]):
    # Can sometimes read pickle3 from python2 by calling twice
    # Can possibly read pickle2 from python3 by using encoding='latin1'
    if isinstance(filepath, str):
        filepath = Path(filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle_to_disk(
    filepath, data
):  # NOTE - cannot pickle lambda or nested functions
    if isinstance(filepath, str):
        filepath = Path(filepath)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_json(filepath: Path):
    assert filepath.exists(), f"Could not find file: {filepath}"
    with open(str(filepath), "r") as f:
        data = json.load(f)
    return data


class JsonBaseModel(BaseModel):
    def to_json(self) -> Dict:
        return self.dict()

    def to_json_file(self, path: Path):
        assert path.parent.exists(), f"Invalid save directory: {path.parent}"
        assert path.suffix == ".json", f"Invalid json file suffix: {path.suffix}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json_file(cls, path: Path) -> BaseModelDerivative:
        with open(path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return cls(**json_dict)


def timeit(func):
    # timer decorator (for debugging only)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def rstatus(message="[bold green] Running simulation..."):
    """
    A decorator that displays a status message while the function runs.

    Args:
        message: The message to display in the status indicator
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with console.Console().status(message) as status:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
