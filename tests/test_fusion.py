import torch

from triad_pcb.fusion import confidence_weights, fuse_maps, top_q_pool


def test_fusion_weights_sum_to_one():
    weights = confidence_weights(0.5, 1.0, 0.5, device=torch.device("cpu"))
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[2] > weights[1]


def test_inactive_branches_receive_zero_weight():
    weights = confidence_weights(
        1.0,
        1.0,
        0.5,
        device=torch.device("cpu"),
        active=(True, False, True),
    )
    assert weights[1] == 0
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_fused_map_and_top_q_are_bounded():
    maps = [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8]), torch.tensor([0.3, 0.7])]
    weights = torch.tensor([0.2, 0.3, 0.5])
    fused = fuse_maps(*maps, weights, residual_weight=0.5)
    assert torch.all((fused >= 0) & (fused <= 1))
    assert torch.isclose(top_q_pool(fused, 0.5), fused.max())
