# pyre-strict
import gc
from typing import Optional
from opensfm import io, reconstruction
from opensfm.dataset_base import DataSetBase


def run_dataset(
    data: DataSetBase, algorithm: Optional[reconstruction.ReconstructionAlgorithm] = None
) -> None:
    """Compute the SfM reconstruction."""

    tracks_manager = data.load_tracks_manager()

    if algorithm is None:
        algo_str = data.config.get("reconstruction_algorithm", "incremental").lower()
        try:
            algorithm = reconstruction.ReconstructionAlgorithm(algo_str.lower())
        except ValueError:
            raise RuntimeError(
                f"Unsupported algorithm for reconstruction '{algo_str}'. Allowed values: {list(v.value for v in reconstruction.ReconstructionAlgorithm)}"
            )

    if algorithm == reconstruction.ReconstructionAlgorithm.INCREMENTAL:
        report, reconstructions = reconstruction.incremental_reconstruction(
            data, tracks_manager
        )
    elif algorithm == reconstruction.ReconstructionAlgorithm.TRIANGULATION:
        report, reconstructions = reconstruction.triangulation_reconstruction(
            data, tracks_manager
        )
    else:
        raise RuntimeError(
            f"Unsupported algorithm for reconstruction {algorithm}")

    del tracks_manager
    gc.collect()

    data.save_report(io.json_dumps(report), "reconstruction.json")
    data.save_reconstruction(reconstructions)
