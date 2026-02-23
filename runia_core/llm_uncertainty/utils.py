from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

from runia_core.import_helper_functions import module_exists

if module_exists("transformers"):
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _are_equivalent(model, tokenizer, text1: str, text2: str) -> bool:
    """
    Determines whether two input texts are semantically equivalent using a
    natural language inference (NLI) model.

    Args:
        model (PreTrainedModel): A HuggingFace model fine-tuned for natural language inference.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer for the model.
        text1 (str): The first input text.
        text2 (str): The second input text.

    Returns:
        bool: True if the texts are semantically equivalent according to the model,
              False otherwise.
    """
    # Forward pass: text1 → text2
    inputs = tokenizer(text1, text2, return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    probs1 = torch.softmax(logits, dim=1)
    result1 = torch.argmax(probs1, dim=1).item()

    # Forward pass: text2 → text1
    inputs = tokenizer(text2, text1, return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    probs2 = torch.softmax(logits, dim=1)
    result2 = torch.argmax(probs2, dim=1).item()

    implications = (result1, result2)
    return (0 not in implications) and (implications != (1, 1))


def _semantic_clustering(model, tokenizer, texts: List[str]) -> Dict[int, List[int]]:
    """
    Clusters input texts into groups of semantically equivalent sentences.

    Args:
        model (PreTrainedModel): A HuggingFace model fine-tuned for natural language inference.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer for the model.
        texts (List[str]): A list of input texts to cluster.

    Returns:
        Dict[int, List[int]]: A dictionary mapping cluster indices to lists of text indices.
                              Each cluster contains the indices of texts that are considered
                              semantically equivalent.
    """
    clusters = []
    clustered_indices = set()

    for i in range(len(texts)):
        if i in clustered_indices:
            continue

        current_cluster = [i]
        clustered_indices.add(i)

        for j in range(i + 1, len(texts)):
            if j in clustered_indices:
                continue

            if _are_equivalent(model, tokenizer, texts[i], texts[j]):
                current_cluster.append(j)
                clustered_indices.add(j)

        clusters.append(current_cluster)

    return {idx: cluster for idx, cluster in enumerate(clusters)}


def _get_probability_distribution(logits: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Converts model logits into probability distributions for each generated token.

    Args:
        logits (Tuple[torch.Tensor, ...]): HuggingFace `outputs.scores`,
                                           where each element is shaped `(batch_size, vocab_size)`
                                           or `(batch_size, 1, vocab_size)`.

    Returns:
        torch.Tensor: Probability distributions of shape `(num_generated, vocab_size)`.
    """
    probs = []
    for logit in logits:
        prob = F.softmax(logit[0], dim=-1)
        probs.append(prob)
    return torch.stack(probs, dim=0).cpu()


def _construct_embedding_matrix(
    hidden_states: Tuple[torch.Tensor, ...], token_index: int = -1, layer_index: int = 15
) -> torch.Tensor:
    """
    Extracts the embedding matrix from the hidden states of a model for EigenScore calculation.

    Args:
        hidden_states (Tuple[torch.Tensor, ...]): HuggingFace `outputs.hidden_states`.
        token_index (int, optional): The index of the token to extract embeddings from.
                                     Defaults to -1 (the last token) which gives the best results according to the EigenScore paper.
        layer_index (int, optional): The index of the layer to extract embeddings from.
                                     Defaults to 15 (middle layer for Llama 2) which gives the best results according to the EigenScore paper.
    Returns:
        torch.Tensor: The embedding matrix of shape `(seq_length, hidden_size)`.
    """
    return hidden_states[token_index][layer_index].squeeze()
