import numpy as np

from triad_pcb.anomaly_ratio_study import stratified_indices


def test_stratified_stream_has_requested_ratio_without_duplicates():
    labels = np.asarray([0] * 700 + [1] * 500)
    indices = stratified_indices(labels, stream_size=500, anomaly_ratio=0.4, seed=17)

    assert len(indices) == 500
    assert len(set(indices)) == 500
    assert labels[indices].sum() == 200
