# pyre-strict
"""Test that feature_type SIFT_GPU dispatches to pypopsift (mocked)."""

import sys
import types
from typing import Any, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import features


def test_sift_gpu_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    sift_gpu_called: bool = False

    def fake_popsift(
        image: NDArray[Any], **kwargs: Any
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        nonlocal sift_gpu_called
        sift_gpu_called = True
        return (
            np.array([[4.0, 4.0, 1.0, 0.0]]),  # one keypoint: x, y, size, angle
            np.array([[1, 2, 3]], dtype=np.uint8),  # its descriptor
        )

    module = types.ModuleType("opensfm.pypopsift")
    module.popsift = fake_popsift  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opensfm.pypopsift", module)

    config = {
        "feature_type": "sift_gpu",  # lowercase on purpose, must be normalized
        "feature_root": False,
        "feature_process_size": 2048,
        "feature_min_frames": 100,
        "sift_edge_threshold": 10,
        "sift_peak_threshold": 0.02,
    }
    image = np.zeros((9, 9, 3), dtype=np.uint8)

    features.extract_features(image, config, False)

    assert sift_gpu_called
