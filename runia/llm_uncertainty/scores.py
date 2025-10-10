"""
Collection of uncertainty scoring functions for LLM outputs.

Each function computes a specific uncertainty metric based on model outputs and intermediate representations.

Requirements:
transformers==4.52.3
"""

import torch
import numpy as np
from typing import Union, List, Dict, Tuple, Any
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from runia.llm_uncertainty.utils import (
    _semantic_clustering,
    _construct_embedding_matrix,
    _get_probability_distribution,
)
from runia.llm_uncertainty.attention_aggregation import (
    _get_recurent_attention,
    _get_average_attention_all,
    _get_attention_rollout,
)

__all__ = [
    "eigen_score",
    "normalized_entropy",
    "semantic_entropy",
    "perplexity",
    "generation_entropy",
    "rauq_uncertainty",
    "rauq_uncertainty_mean_heads",
    "rauq_uncertainty_rollout",
    "RAUQ",
    "compute_uncertainties",
]


def eigen_score(hidden_states: Tuple[torch.Tensor, ...], alpha: float = 1e-3) -> float:
    """
    Computes the average log of singular values of the covariance matrix of embeddings,
    with a small regularization term added for numerical stability. Introduced by Chen et al. 2024 https://arxiv.org/abs/2402.03744

    Args:
        embedding_matrix (torch.Tensor): Tensor of shape (num_samples, num_hidden_states)
                                         representing embeddings of samples.
        alpha (float, optional): Small regularization constant added to the diagonal of
                                 the covariance matrix. Defaults to 1e-3.

    Returns:
        float: Mean of the logarithm of the singular values of the covariance matrix.
    """
    embedding_matrix = _construct_embedding_matrix(hidden_states)
    cov_matrix = torch.cov(embedding_matrix.T).cpu().numpy().astype(float)
    _, s, _ = np.linalg.svd(cov_matrix + alpha * np.eye(cov_matrix.shape[0]))
    return float(np.mean(np.log(s)))


def normalized_entropy(log_probs: torch.Tensor) -> float:
    """
    Computes the normalized entropy across sequences of log probabilities. Introduced by Malinin and Gales 2021 https://arxiv.org/abs/2002.07650

    Args:
        log_probs (torch.Tensor): Tensor of shape `(num_sequences, seq_len)` containing
                                  log probabilities of output tokens in each sequence.

    Returns:
        float: The normalized entropy across all sequences.
    """
    n = len(log_probs)
    entropy = 0.0
    for seq in log_probs:
        valid = seq != -float("inf")
        entropy += torch.sum(seq[valid]) / torch.sum(valid)
    return (-entropy / n).item()


def semantic_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> Tuple[float, Dict[int, List[int]]]:
    """
    Computes discrete semantic entropy by combining clustering of semantically equivalent texts
    with probability-based entropy from model outputs. Introduced by Kuhn, Gal, and Farquhar 2023 https://arxiv.org/abs/2302.09664

    Args:
        model (PreTrainedModel): A HuggingFace model fine-tuned for natural language inference,
                                 used for clustering texts.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        texts (List[str]): List of input texts to cluster.

    Returns:
        Tuple[float, Dict[int, List[int]]]:
            - `discrete_semantic_entropy`: Entropy based on cluster size distribution.
            - `clusters`: Dictionary mapping cluster indices to lists of text indices.
    """
    clusters = _semantic_clustering(model, tokenizer, texts)

    # Entropy based on cluster sizes
    total_samples = sum(len(indices) for indices in clusters.values())
    discrete_semantic_entropy = 0.0
    for indices in clusters.values():
        cluster_prob = len(indices) / total_samples
        if cluster_prob > 0:
            discrete_semantic_entropy -= cluster_prob * np.log(cluster_prob)

    return discrete_semantic_entropy, clusters


def perplexity(log_probs: List[float]) -> float:
    """
    Computes the perplexity of a sequence based on its log probabilities.

    Args:
        log_probs (torch.Tensor): Tensor of shape `(seq_len,)` containing log probabilities
                                  of the predicted output tokens.

    Returns:
        float: The perplexity of the sequence.
    """
    return -torch.mean(log_probs).item()


def generation_entropy(logits: torch.FloatTensor) -> float:
    """
    Computes the entropy averaged across generated tokens.

    Args:
        logits (torch.FloatTensor): HuggingFace `outputs.scores`,
                                    where each element is shaped `(batch_size, seq_len, vocab_size)`.

    Returns:
        float: Average entropy entropy across all generated tokens.
    """
    prob_dist = _get_probability_distribution(logits)
    entropies = []
    for p in prob_dist:
        log_p = torch.clamp(p, min=1e-12).log()
        entropy = -(p * log_p).sum() / torch.log(torch.tensor(p.shape[-1], dtype=torch.float32))
        entropies.append(entropy.item())
    return float(np.mean(entropies))


def rauq_uncertainty(
    log_probs: torch.Tensor,
    attentions: Tuple[Tuple[torch.Tensor, ...], ...],
    token_aggregation: str,
    alphas: List[float] = [0.2],
    ablation: bool = False,
) -> Union[float, List[float]]:
    """
    Computes the uncertainty score based on the original RAUQ (Recurrent Attention-based Uncertainty Quantification) algorithm.
    Introduced by Vazhentsev et al. 2025 https://arxiv.org/abs/2505.20045

    Args:
        log_probs (torch.Tensor): Tensor of shape `(seq_len,)` containing log probabilities
                                  of the predicted output tokens.
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        token_aggregation (str): Method to aggregate attention over tokens.
                                 Options are 'original' (attention to the previous token)
                                 or 'mean_all_tokens' (average attention over all past tokens).
        alphas (List[float], optional): List of hyperparameter values controlling the
                                        trade-off between token probability and propagated confidence from the
                                        previous token. Defaults to `[0.2]`.
        ablation (bool, optional): If True, returns a list of uncertainty scores for all
                                   `alphas`. If False, returns only the first score.
                                   Defaults to False.

    Returns:
        Union[float, List[float]]:
            - If `ablation` is False: the RAUQ uncertainty score for the first alpha.
            - If `ablation` is True: a list of RAUQ uncertainty scores corresponding to
              each alpha value in `alphas`.
    """
    aggregate_tokens = {
        "original": _get_recurent_attention,
        "mean_all_tokens": _get_average_attention_all,
    }
    attention_weights = aggregate_tokens[token_aggregation](attentions)
    L, _, N = attention_weights.shape

    head_l = []
    for layer in range(L):
        avg_att = attention_weights[layer, :, 1:].mean(dim=1)
        head_idx = torch.argmax(avg_att).item()
        head_l.append(head_idx)

    probs = log_probs.exp().squeeze()

    uncertainty_alpha = []
    for alpha in alphas:
        confidence_l = torch.zeros((N, L))
        confidence_l[0, :] = probs[0] if probs.dim() > 0 else probs.item()
        for i in range(1, N):
            att = torch.stack(
                [
                    attention_weights[layer, head_l[layer], i]
                    for layer in range(attention_weights.shape[0])
                ]
            )
            confidence_l[i, :] = (
                alpha * probs[i] * torch.ones_like(confidence_l[i - 1, :])
                + (1 - alpha) * att * confidence_l[i - 1, :]
            )
        uncertainty_l = -torch.mean(torch.log(confidence_l), dim=0)
        uncertainty_alpha.append(uncertainty_l.max().item())

    return uncertainty_alpha[0] if not ablation else uncertainty_alpha


def rauq_uncertainty_mean_heads(
    log_probs: torch.Tensor,
    attentions: Tuple[Tuple[torch.Tensor, ...], ...],
    token_aggregation: str,
    alphas: List[float] = [0.3],
    ablation: bool = False,
) -> Union[float, List[float]]:
    """
    Computes the RAUQ uncertainty score by averaging attention over all heads in each layer.

    Args:
        log_probs (torch.Tensor): Tensor of shape `(seq_len,)` containing log probabilities
                                  of the predicted output tokens.
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        token_aggregation (str): Method to aggregate attention over tokens.
                                 Options are 'original' (attention to the previous token)
                                 or 'mean_all_tokens' (average attention over all past tokens).
        alphas (List[float], optional): List of hyperparameter values controlling the balance
                                        between token probability and attention propagation.
                                        Defaults to [0.3].
        ablation (bool, optional): If True, returns uncertainty scores for all `alphas`. If False,
                                   returns only the first score. Defaults to False.

    Returns:
        Union[float, List[float]]: RAUQ uncertainty score(s), depending on `ablation`.
    """
    aggregate_tokens = {
        "original": _get_recurent_attention,
        "mean_all_tokens": _get_average_attention_all,
    }
    attention_weights = aggregate_tokens[token_aggregation](attentions)
    L, _, N = attention_weights.shape

    # Average over attention heads
    attention_weights = attention_weights.mean(dim=1)

    probs = log_probs.exp().squeeze()

    uncertainty_alpha = []
    for alpha in alphas:
        confidence_l = torch.zeros((N, L))
        confidence_l[0, :] = probs[0] if probs.dim() > 0 else probs.item()
        for i in range(1, N):
            att = attention_weights[:, i]
            confidence_l[i, :] = (
                alpha * probs[i] * torch.ones_like(confidence_l[i - 1, :])
                + (1 - alpha) * att * confidence_l[i - 1, :]
            )
        uncertainty_l = -torch.mean(torch.log(confidence_l), dim=0)
        uncertainty_alpha.append(uncertainty_l.max().item())

    return uncertainty_alpha[0] if not ablation else uncertainty_alpha


def rauq_uncertainty_rollout(
    log_probs: torch.Tensor,
    attentions: Tuple[Tuple[torch.Tensor, ...], ...],
    token_aggregation: str,
    input_length: int,
    alphas: List[float] = [0.4],
    ablation: bool = False,
) -> Union[float, List[float]]:
    """
    Computes the RAUQ uncertainty score using the attention rollout matrix.

    Args:
        log_probs (torch.Tensor): Tensor of shape `(seq_len,)` containing log probabilities
                                  of the predicted output tokens.
        attentions (Tuple[Tuple[torch.Tensor, ...], ...]): Model attention weights from HuggingFace outputs.
        token_aggregation (str): Method to aggregate attention over tokens.
                                 Options are 'original' (attention to the previous token)
                                 or 'mean_all_tokens' (average attention over all past tokens).
        input_length (int): Length of the original input sequence.
        alphas (List[float], optional): List of hyperparameter values controlling the balance
                                        between token probability and attention propagation.
                                        Defaults to [0.4].
        ablation (bool, optional): If True, returns uncertainty scores for all `alphas`. If False,
                                   returns only the first score. Defaults to False.

    Returns:
        Union[float, List[float]]: RAUQ uncertainty score(s), depending on `ablation`.
    """
    attention_rollout = _get_attention_rollout(attentions, input_length)
    if token_aggregation == "original":
        attention_weights = attention_rollout.diagonal(offset=-1)[-log_probs.shape[1] :]
    elif token_aggregation == "mean_all_tokens":
        attention_weights = attention_rollout[:, -log_probs.shape[1] :].mean(dim=0)
    probs = log_probs.exp().squeeze()
    N = probs.shape[0]

    uncertainty_alpha = []
    for alpha in alphas:
        confidence_l = torch.zeros((N,))
        confidence_l[0] = probs[0] if probs.dim() > 0 else probs.item()
        for i in range(1, N):
            att = attention_weights[i]
            confidence_l[i] = alpha * probs[i] + (1 - alpha) * att * confidence_l[i - 1]
        uncertainty_l = -torch.mean(torch.log(confidence_l))
        uncertainty_alpha.append(uncertainty_l.item())

    return uncertainty_alpha if ablation else uncertainty_alpha[0]


def RAUQ(
    log_probs, attentions, input_length, token_aggregation, head_aggregation, alphas, ablation
):
    rauq_functions = {
        "original": lambda log_probs, attentions, token_aggregation, input_length, alphas, ablation: (
            rauq_uncertainty(log_probs, attentions, token_aggregation, alphas, ablation)
        ),
        "mean_heads": lambda log_probs, attentions, token_aggregation, input_length, alphas, ablation: (
            rauq_uncertainty_mean_heads(log_probs, attentions, token_aggregation, alphas, ablation)
        ),
        "rollout": lambda log_probs, attentions, token_aggregation, input_length, alphas, ablation: (
            rauq_uncertainty_rollout(
                log_probs, attentions, token_aggregation, input_length, alphas, ablation
            )
        ),
    }

    return rauq_functions[head_aggregation](
        log_probs, attentions, token_aggregation, input_length, alphas, ablation
    )


def compute_uncertainties(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    uncertainty_requests: List[Dict[str, Any]],
    gen_config: GenerationConfig = None,
    num_samples: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text and compute requested uncertainty scores.

    Args:
        model (PreTrainedModel): HuggingFace model.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
        gen_config (GenerationConfig): Generation parameters.
        prompt (str): Input text.
        uncertainty_requests (List[Dict[str, Any]]): Each dict has at least {"method_name": str}.
                                                     For RAUQ also needs {"token_aggregation", "head_aggregation", "alphas", "ablation"}.
                                                     "method_name" options:
                                                     - "eigen_score"
                                                     - "normalized_entropy"
                                                     - "semantic_entropy"
                                                     - "perplexity"
                                                     - "generation_entropy"
                                                     - "RAUQ"
                                                     "token_aggregation" options for RAUQ:
                                                     - "original"
                                                     - "mean_all_tokens"
                                                     "head_aggregation" options for RAUQ:
                                                     - "original"
                                                     - "mean_heads"
                                                     - "rollout"
                                                     "alphas": List of float in [0,1] hyperparameters for RAUQ.
                                                     "ablation": bool, if True returns scores for all alphas.
        num_samples (int, optional): Number of samples if multiple generations required.

    Returns:
        Tuple[str, Dict[str, Any]]:
            - Generated text (from deterministic generation).
            - Dict mapping uncertainty function name to score/output.
    """

    registry = {
        "eigen_score": {
            "fn": lambda det, samp, req: eigen_score(samp["hidden_states"]),
            "needs_sampling": True,
        },
        "normalized_entropy": {
            "fn": lambda det, samp, req: normalized_entropy(samp["log_probs"]),
            "needs_sampling": True,
        },
        "semantic_entropy": {
            "fn": lambda det, samp, req: semantic_entropy(
                samp["model_entailment"], samp["tokenizer_entailment"], samp["texts"]
            ),
            "needs_sampling": True,
        },
        "perplexity": {
            "fn": lambda det, samp, req: perplexity(det["log_probs"]),
            "needs_sampling": False,
        },
        "generation_entropy": {
            "fn": lambda det, samp, req: generation_entropy(det["logits"]),
            "needs_sampling": False,
        },
        "RAUQ": {
            "fn": lambda det, samp, req: RAUQ(
                det["log_probs"],
                det["attentions"],
                input_length=det["input_length"],
                token_aggregation=req.get("token_aggregation", "mean_all_tokens"),
                head_aggregation=req.get("head_aggregation", "rollout"),
                alphas=req.get("alphas", [0.3]),
                ablation=req.get("ablation", False),
            ),
            "needs_sampling": False,
        },
    }

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    need_semantic_clustering = any(
        req["method_name"] == "semantic_entropy" for req in uncertainty_requests
    )
    if need_semantic_clustering:
        model_entailment = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xxlarge-mnli", device_map="auto"
        )
        tokenizer_entailment = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")

    # --- Step 1: Deterministic generation ---
    det_out = model.generate(
        **inputs,
        generation_config=gen_config,
        output_attentions=True,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        tokenizer=tokenizer,
    )
    deterministic_text = tokenizer.batch_decode(
        det_out.sequences[:, input_length:], skip_special_tokens=True
    )

    det_log_probs = model.compute_transition_scores(
        det_out.sequences,
        det_out.scores,
        normalize_logits=True,
    )

    deterministic = {
        "log_probs": det_log_probs,
        "logits": det_out.scores,
        "attentions": det_out.attentions,
        "input_length": input_length,
        "text": deterministic_text,
    }

    # --- Step 2: Sampled generations if needed ---
    needs_sampling = any(
        registry[req["method_name"]]["needs_sampling"] for req in uncertainty_requests
    )

    sampled = {
        "log_probs": None,
        "hidden_states": None,
        "texts": None,
        "model_entailment": model_entailment if need_semantic_clustering else None,
        "tokenizer_entailment": tokenizer_entailment if need_semantic_clustering else None,
    }
    if needs_sampling:
        samp_out = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=num_samples,
            generation_config=gen_config,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sampled_texts = tokenizer.batch_decode(
            samp_out.sequences[:, input_length:], skip_special_tokens=True
        )

        sampled = {
            "log_probs": model.compute_transition_scores(
                samp_out.sequences,
                samp_out.scores,
                normalize_logits=True,
            ),
            "hidden_states": samp_out.hidden_states,
            "texts": sampled_texts,
            "model_entailment": model_entailment if need_semantic_clustering else None,
            "tokenizer_entailment": tokenizer_entailment if need_semantic_clustering else None,
        }

    # --- Step 3: Compute requested scores ---
    scores = {}
    for req in uncertainty_requests:
        name = (
            req["method_name"]
            + ("_" + str(req["token_aggregation"]) if req["method_name"] == "RAUQ" else "")
            + ("_" + str(req["head_aggregation"]) if req["method_name"] == "RAUQ" else "")
        )
        fn_entry = registry[req["method_name"]]
        scores[name] = fn_entry["fn"](deterministic, sampled, req)
        if req["method_name"] == "semantic_entropy":
            scores["clusters"] = {
                sampled["texts"][i]: cluster
                for cluster, texts in scores[name][1].items()
                for i in texts
            }
            scores[name] = scores[name][0]

    return deterministic_text, scores
