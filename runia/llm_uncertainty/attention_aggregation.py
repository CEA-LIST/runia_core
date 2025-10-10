import torch
from typing import Tuple


def reconstruct_attention_matrix(
    attentions: Tuple[Tuple[torch.Tensor, ...], ...], input_length: int
) -> torch.Tensor:
    """
    Reconstructs a full attention map across all generated tokens, heads and layers.

    This function rebuilds the full attention history from HuggingFace
    `outputs.attentions`, which is structured as:
        - a tuple for each generated token,
        - containing a tuple for each layer,
        - with attention tensors (typically shaped `(batch_size, num_heads, tgt_len, src_len)`).

    Args:
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        input_length (int): Length of the original input sequence.

    Returns:
        torch.Tensor: Reconstructed attention map of shape
                      `(num_layers, num_heads, total_seq_len, total_seq_len)`,
                      where `total_seq_len = input_length + num_generated`.
    """
    num_generated = len(attentions)
    num_layers = len(attentions[0])
    batch_size, num_heads = attentions[0][0].shape[:2]
    total_seq_len = input_length + num_generated

    full_attentions = torch.zeros((num_layers, batch_size, num_heads, total_seq_len, total_seq_len))

    for generated_idx, per_layer_attn in enumerate(attentions):
        for layer_idx, attn in enumerate(per_layer_attn):
            if generated_idx == 0:
                # For some models (e.g., Gemma), the slice may differ
                full_attentions[layer_idx, :, :, :input_length, :input_length] = attn
            else:
                # For Gemma: attn[:, :, 0, :input_length + generated_idx]
                full_attentions[
                    layer_idx, :, :, input_length + generated_idx, : input_length + generated_idx
                ] = attn.squeeze(2)

    return full_attentions.squeeze(1).cpu()


def get_attention_rollout(
    attentions: Tuple[Tuple[torch.Tensor, ...], ...], input_length: int
) -> torch.Tensor:
    """
    Computes attention rollout, aggregating attention information across layers.

    Based on the method from Abnar & Zuidema (2020) https://arxiv.org/abs/2005.00928, this performs a recursive
    multiplication of attention matrices with an identity matrix added,
    to propagate influence scores through layers.

    Args:
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        input_length (int): Length of the original input sequence.

    Returns:
        torch.Tensor: Joint attention matrix of shape `(seq_len, seq_len)`.
    """
    attn = reconstruct_attention_matrix(attentions, input_length)
    L, H, N, _ = attn.shape
    identity = torch.eye(N, device=attn.device)

    augmented_attn = []
    for l in range(L):
        A = attn[l].mean(dim=0) + identity
        A = A / A.sum(dim=-1, keepdim=True)
        augmented_attn.append(A)

    joint = augmented_attn[0]
    for l in range(1, L):
        joint = augmented_attn[l] @ joint

    return joint


def get_recurent_attention(
    attentions: Tuple[Tuple[torch.Tensor, ...], ...], position: int = 1
) -> torch.Tensor:
    """
    Extracts recurrent attention weights at a given relative position.

    Specifically, retrieves attention weights that each generated token assigns
    to a preceding token (e.g., `position=1` means the immediate previous token).

    Args:
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        position (int): Relative position of the attended token (e.g., 1 = previous token).

    Returns:
        torch.Tensor: Tensor of shape `(num_layers, num_heads, num_generated-1)`.
    """
    num_generated = len(attentions)
    num_layers = len(attentions[0])
    _, num_heads = attentions[0][0].shape[:2]

    full_attentions = torch.zeros((num_layers, num_heads, num_generated - 1))

    for generated_idx, per_layer_attn in enumerate(attentions[1:]):
        for layer_idx, attn in enumerate(per_layer_attn):
            full_attentions[layer_idx, :, generated_idx] = attn[0, :, 0, -position - 1]

    return full_attentions.cpu()


def get_average_attention_all(
    attentions: Tuple[Tuple[torch.Tensor, ...], ...],
) -> torch.Tensor:
    """
    Computes the average attention weights over all past tokens for each generated token.

    Args:
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.

    Returns:
        torch.Tensor: Tensor of shape `(num_layers, num_heads, num_generated)`,
                      containing averaged attention values.
    """
    num_generated = len(attentions)
    num_layers = len(attentions[0])
    _, num_heads = attentions[0][0].shape[:2]

    average_attention = torch.zeros((num_layers, num_heads, num_generated))

    for generated_idx, per_layer_attn in enumerate(attentions):
        for layer_idx, attn in enumerate(per_layer_attn):
            # For Gemma: attn[0, :, :, input_length + generated_idx - 1].mean(dim=1)
            average_attention[layer_idx, :, generated_idx] = attn[0, :, 0, :].mean(dim=1)

    return average_attention
