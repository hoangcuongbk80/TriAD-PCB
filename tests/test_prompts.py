import pytest
import torch

from triad_pcb.prompts import SemanticPromptBank


def test_prompt_context_shapes_and_gradients():
    anchors = torch.eye(3, 4)
    normal = torch.eye(3, 4)
    abnormal = torch.randn(3, 2, 4)
    bank = SemanticPromptBank(anchors, normal, abnormal, context_length=5)
    learned_anchors, learned_normal, learned_abnormal = bank.normalized()

    assert learned_anchors.shape == (3, 4)
    assert learned_normal.shape == (3, 4)
    assert learned_abnormal.shape == (3, 2, 4)
    assert bank.normal_context.shape == (3, 5, 4)
    assert bank.abnormal_context.shape == (3, 2, 5, 4)
    (learned_normal.sum() + learned_abnormal.sum()).backward()
    assert bank.normal_context.grad is not None
    assert bank.abnormal_context.grad is not None


def test_prompt_context_length_must_be_positive():
    anchors = torch.eye(2)
    abnormal = torch.randn(2, 2, 2)
    with pytest.raises(ValueError, match="positive"):
        SemanticPromptBank(anchors, anchors, abnormal, context_length=0)
