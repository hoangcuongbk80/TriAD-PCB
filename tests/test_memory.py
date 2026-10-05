import torch

from triad_pcb.engine import CausalInferenceSession
from triad_pcb.memory import HistoricalBank, redundancy_aware_sample


def test_ras_keeps_early_unique_features():
    features = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    sampled = redundancy_aware_sample(features, 2)
    assert torch.equal(sampled, features[[0, 2]])


def test_history_is_bounded_and_scores_are_valid():
    bank = HistoricalBank(capacity=2, sampling_ratio=0.5, top_ratio=0.5)
    bank.update(torch.eye(4))
    bank.update(torch.roll(torch.eye(4), 1, 0))
    bank.update(torch.roll(torch.eye(4), 2, 0))
    scores = bank.score(torch.eye(4))
    assert len(bank) == 2
    assert torch.all((scores >= 0) & (scores <= 1))


class DummyModel:
    def eval(self):
        return self

    def encode_image(self, images):
        patches = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-0.7, 0.7]]], dtype=images.dtype
        ).expand(images.shape[0], -1, -1)
        global_features = torch.tensor([[1.0, 0.0]], dtype=images.dtype).expand(images.shape[0], -1)
        return patches, global_features

    def text_map(self, patches, global_features):
        return torch.zeros(patches.shape[:2]), torch.ones((patches.shape[0], 1))


def test_causal_session_scores_before_history_update():
    config = {
        "memory": {
            "history_capacity": 2,
            "sampling_ratio": 0.5,
            "history_top_ratio": 0.5,
            "normality_threshold": 1.0,
            "minimum_reference_patches": 1,
        },
        "model": {
            "fusion_temperature": 0.5,
            "residual_weight": 0.5,
            "top_q": 0.5,
        },
    }
    session = CausalInferenceSession(DummyModel(), config)
    image = torch.zeros((1, 3, 2, 2))
    session.initialize_support(image)
    first = session.step(image)
    second = session.step(image)
    assert first["history_size_after_update"] == 1
    assert second["history_size_after_update"] == 2
    assert second["fusion_weights"][1] > first["fusion_weights"][1]
