import torch

from triad_pcb.validation import localized_validation_anomaly


def test_validation_anomaly_is_deterministic_and_localized():
    image = torch.arange(3 * 64 * 64, dtype=torch.float32).reshape(3, 64, 64)
    first, first_mask, first_name = localized_validation_anomaly(image, 3407)
    second, second_mask, second_name = localized_validation_anomaly(image, 3407)
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert first_name == second_name
    affected_fraction = float(first_mask.mean())
    assert 0.15 < affected_fraction < 0.30
    assert torch.any(first != image)
