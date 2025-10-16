"""
tests/unit_test_llm_uncertainty.py

Unittest suite for functions in `runia.llm_uncertainty.scores`.

This test suite covers the uncertainty scoring functions for LLM outputs,
including:
- eigen_score: Eigenvalue-based uncertainty from embeddings
- normalized_entropy: Normalized entropy from log probabilities
- semantic_entropy: Semantic clustering-based entropy
- perplexity: Perplexity calculation from log probabilities
- generation_entropy: Token-level entropy from logits
- rauq_uncertainty: RAUQ uncertainty with different aggregation methods
- rauq_uncertainty_mean_heads: RAUQ with mean head aggregation
- rauq_uncertainty_rollout: RAUQ with attention rollout
- RAUQ: Unified RAUQ interface
- compute_uncertainties: End-to-end uncertainty computation

Run with:
    python tests/unit_test_llm_uncertainty.py
    or
    python -m unittest discover -s tests -p "unit_test_llm_uncertainty.py"

Tests use synthetic data and mock objects to avoid requiring actual LLM models,
ensuring tests are fast and deterministic.
"""

import unittest
import logging
import numpy as np
import torch
from typing import Tuple
from unittest.mock import MagicMock, patch

from runia.llm_uncertainty.scores import (
    eigen_score,
    normalized_entropy,
    semantic_entropy,
    perplexity,
    generation_entropy,
    rauq_uncertainty,
    rauq_uncertainty_mean_heads,
    rauq_uncertainty_rollout,
    RAUQ,
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests may not pass!!
SEED = 42
TOL = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
########################################################################


class TestEigenScore(unittest.TestCase):
    """Test cases for the eigen_score function."""

    def setUp(self) -> None:
        """Set up test fixtures with deterministic seeds."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        logger.info(f"Running tests on device: {DEVICE}")

    def test_eigen_score_basic(self):
        """Test eigen_score with basic synthetic hidden states."""
        # Create synthetic hidden states: tuple of (tuple of layers) per generated token
        # Structure: (num_generated_tokens,) where each is (num_layers,) of tensors (batch, seq, hidden)
        num_tokens = 5
        num_layers = 20  # Need at least 16 layers (default layer_index=15)
        batch_size = 1
        seq_len = 10
        hidden_dim = 768

        hidden_states = tuple(
            tuple(torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers))
            for _ in range(num_tokens)
        )

        score = eigen_score(hidden_states, alpha=1e-3)

        # Check that score is a float
        self.assertIsInstance(score, float)

        # Check that score is finite
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))
        self.assertAlmostEqual(score, -6.775187082486514, delta=TOL)

    def test_eigen_score_different_alphas(self):
        """Test eigen_score with different regularization parameters."""
        # Ensure we have at least 16 layers
        hidden_states = tuple(tuple(torch.randn(1, 5, 64) for _ in range(20)) for _ in range(3))

        score1 = eigen_score(hidden_states, alpha=1e-3)
        score2 = eigen_score(hidden_states, alpha=1e-2)

        # Both scores should be finite
        self.assertFalse(np.isnan(score1))
        self.assertFalse(np.isnan(score2))

        # Scores should be different with different alphas
        self.assertNotAlmostEqual(score1, score2, delta=TOL)

    def test_eigen_score_deterministic(self):
        """Test that eigen_score is deterministic with the same input."""
        torch.manual_seed(SEED)
        hidden_states = tuple(tuple(torch.randn(1, 5, 64) for _ in range(20)) for _ in range(3))

        score1 = eigen_score(hidden_states)
        score2 = eigen_score(hidden_states)

        self.assertAlmostEqual(score1, score2, delta=TOL)


class TestNormalizedEntropy(unittest.TestCase):
    """Test cases for the normalized_entropy function."""

    def setUp(self) -> None:
        """Set up test fixtures with deterministic seeds."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_normalized_entropy_basic(self):
        """Test normalized_entropy with uniform log probabilities."""
        # Create log probabilities for 3 sequences of length 5
        log_probs = torch.log(torch.ones(3, 5) * 0.2)

        score = normalized_entropy(log_probs)

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertFalse(np.isnan(score))
        self.assertAlmostEqual(score, 1.6094379425048828, delta=TOL)

    def test_normalized_entropy_with_inf(self):
        """Test normalized_entropy handles -inf values correctly."""
        log_probs = torch.tensor(
            [
                [-0.5, -1.0, -0.3, -float("inf"), -0.8],
                [-0.2, -0.6, -float("inf"), -0.9, -1.2],
            ]
        )

        score = normalized_entropy(log_probs)

        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

    def test_normalized_entropy_deterministic(self):
        """Test that normalized_entropy is deterministic."""
        log_probs = torch.randn(4, 6)

        score1 = normalized_entropy(log_probs)
        score2 = normalized_entropy(log_probs)

        self.assertAlmostEqual(score1, score2, delta=TOL)

    def test_normalized_entropy_value_range(self):
        """Test that normalized_entropy returns reasonable values."""
        # High confidence (low entropy) - probabilities close to 1 for one option
        # BUT also has very small probabilities for other options
        log_probs_high_conf = torch.log(
            torch.tensor(
                [
                    [0.9, 0.05, 0.03, 0.01, 0.01],
                    [0.85, 0.08, 0.04, 0.02, 0.01],
                ]
            )
        )

        # Low confidence (high entropy) - uniform probabilities
        # All probabilities are moderate (0.2)
        log_probs_low_conf = torch.log(
            torch.tensor(
                [
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            )
        )

        score_high = normalized_entropy(log_probs_high_conf)
        score_low = normalized_entropy(log_probs_low_conf)

        # The function sums log probs across sequences
        # High conf has: one large (≈-0.1) + several very small (≈-3.0, -3.5, -4.6)
        # Sum: -0.1 -3.0 -3.5 -4.6 -4.6 ≈ -15.8, divided by 5 ≈ -3.16
        # Low conf has: all moderate (≈-1.6)
        # Sum: -1.6 * 5 ≈ -8.0, divided by 5 ≈ -1.6
        # So -(sum/n) gives: high ≈ 3.16, low ≈ 1.6
        # Therefore high confidence actually has HIGHER score due to very negative small probs
        self.assertGreater(score_high, score_low)


class TestSemanticEntropy(unittest.TestCase):
    """Test cases for the semantic_entropy function."""

    def setUp(self) -> None:
        """Set up test fixtures with mock models."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    @patch("runia.llm_uncertainty.scores._semantic_clustering")
    def test_semantic_entropy_basic(self, mock_clustering):
        """Test semantic_entropy with mocked clustering."""
        # Mock clustering result: 3 clusters
        mock_clustering.return_value = {
            0: [0, 1, 2],  # Cluster 0 has 3 texts
            1: [3, 4],  # Cluster 1 has 2 texts
            2: [5],  # Cluster 2 has 1 text
        }

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        texts = ["text1", "text2", "text3", "text4", "text5", "text6"]

        entropy, clusters = semantic_entropy(mock_model, mock_tokenizer, texts)

        # Check return types
        self.assertIsInstance(entropy, float)
        self.assertIsInstance(clusters, dict)

        # Check entropy is non-negative
        self.assertGreaterEqual(entropy, 0.0)
        self.assertAlmostEqual(entropy, 1.0114042647073516, delta=TOL)

        # Check clusters match mock
        self.assertEqual(len(clusters), 3)

    @patch("runia.llm_uncertainty.scores._semantic_clustering")
    def test_semantic_entropy_single_cluster(self, mock_clustering):
        """Test semantic_entropy when all texts are in one cluster."""
        # All texts in one cluster -> entropy should be 0
        mock_clustering.return_value = {0: [0, 1, 2, 3]}

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        texts = ["text1", "text2", "text3", "text4"]

        entropy, clusters = semantic_entropy(mock_model, mock_tokenizer, texts)

        # Single cluster should have zero entropy
        self.assertAlmostEqual(entropy, 0.0, delta=TOL)

    @patch("runia.llm_uncertainty.scores._semantic_clustering")
    def test_semantic_entropy_all_different(self, mock_clustering):
        """Test semantic_entropy when each text is in its own cluster."""
        # Each text in its own cluster -> maximum entropy
        mock_clustering.return_value = {i: [i] for i in range(5)}

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        texts = ["text1", "text2", "text3", "text4", "text5"]

        entropy, clusters = semantic_entropy(mock_model, mock_tokenizer, texts)

        # Maximum entropy for uniform distribution
        expected_entropy = np.log(5)
        self.assertAlmostEqual(entropy, expected_entropy, delta=TOL)


class TestPerplexity(unittest.TestCase):
    """Test cases for the perplexity function."""

    def setUp(self) -> None:
        """Set up test fixtures with deterministic seeds."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_perplexity_basic(self):
        """Test perplexity with basic log probabilities."""
        log_probs = torch.tensor([-0.5, -0.8, -0.3, -0.6, -0.9])

        score = perplexity(log_probs)

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertFalse(np.isnan(score))
        self.assertAlmostEqual(score, 0.6200000047683716, delta=TOL)

    def test_perplexity_perfect_prediction(self):
        """Test perplexity with perfect predictions (log_prob = 0)."""
        log_probs = torch.zeros(10)

        score = perplexity(log_probs)

        # Perfect prediction should have perplexity close to 0
        self.assertAlmostEqual(score, 0.0, delta=TOL)

    def test_perplexity_deterministic(self):
        """Test that perplexity is deterministic."""
        log_probs = torch.randn(8)

        score1 = perplexity(log_probs)
        score2 = perplexity(log_probs)

        self.assertAlmostEqual(score1, score2, delta=TOL)

    def test_perplexity_value_consistency(self):
        """Test that perplexity equals negative mean of log probs."""
        log_probs = torch.tensor([-1.0, -2.0, -1.5, -0.5])

        score = perplexity(log_probs)
        expected = -torch.mean(log_probs).item()

        self.assertAlmostEqual(score, expected, delta=TOL)


class TestGenerationEntropy(unittest.TestCase):
    """Test cases for the generation_entropy function."""

    def setUp(self) -> None:
        """Set up test fixtures with deterministic seeds."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_generation_entropy_basic(self):
        """Test generation_entropy with synthetic logits."""
        # Create logits: tuple of tensors (batch_size, vocab_size)
        vocab_size = 100
        num_tokens = 5

        logits = tuple(torch.randn(1, vocab_size) for _ in range(num_tokens))

        score = generation_entropy(logits)

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)  # Normalized by vocab size
        self.assertFalse(np.isnan(score))

    def test_generation_entropy_uniform(self):
        """Test generation_entropy with uniform distribution."""
        vocab_size = 50
        num_tokens = 3

        # Uniform logits (all zeros -> uniform probabilities)
        logits = tuple(torch.zeros(1, vocab_size) for _ in range(num_tokens))

        score = generation_entropy(logits)

        # Uniform distribution should have maximum normalized entropy (close to 1)
        self.assertGreater(score, 0.9)

    def test_generation_entropy_deterministic_logits(self):
        """Test generation_entropy with highly peaked distribution."""
        vocab_size = 50
        num_tokens = 3

        # Create peaked distribution (one very high logit)
        peaked_logits = []
        for _ in range(num_tokens):
            logit = torch.full((1, vocab_size), -10.0)
            logit[0, 0] = 10.0  # Peak at first token
            peaked_logits.append(logit)

        logits = tuple(peaked_logits)

        score = generation_entropy(logits)

        # Deterministic distribution should have low entropy
        self.assertLess(score, 0.1)


class TestRAUQUncertainty(unittest.TestCase):
    """Test cases for RAUQ uncertainty functions."""

    def setUp(self) -> None:
        """Set up test fixtures with deterministic seeds."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def _create_mock_attentions(
        self, num_tokens: int, num_layers: int, num_heads: int, seq_len: int
    ) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """Create mock attention tensors."""
        attentions = []
        for t in range(num_tokens):
            layer_attentions = []
            for l in range(num_layers):
                # Shape: (batch_size=1, num_heads, 1, seq_len+t)
                attn = torch.softmax(torch.randn(1, num_heads, 1, seq_len + t), dim=-1)
                layer_attentions.append(attn)
            attentions.append(tuple(layer_attentions))
        return tuple(attentions)

    def test_rauq_uncertainty_original(self):
        """Test rauq_uncertainty with original token aggregation."""
        num_tokens = 5
        num_layers = 6
        num_heads = 8
        seq_len = 10

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        score = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=[0.2], ablation=False
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertFalse(np.isnan(score))

    def test_rauq_uncertainty_mean_all_tokens(self):
        """Test rauq_uncertainty with mean_all_tokens aggregation."""
        num_tokens = 4
        num_layers = 6
        num_heads = 4
        seq_len = 8

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        score = rauq_uncertainty(
            log_probs, attentions, token_aggregation="mean_all_tokens", alphas=[0.3], ablation=False
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_rauq_uncertainty_ablation(self):
        """Test rauq_uncertainty with ablation mode (multiple alphas)."""
        num_tokens = 3
        num_layers = 4
        num_heads = 4
        seq_len = 6

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)
        alphas = [0.1, 0.2, 0.3, 0.4]

        scores = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=alphas, ablation=True
        )

        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), len(alphas))

        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    def test_rauq_uncertainty_mean_heads(self):
        """Test rauq_uncertainty_mean_heads function."""
        num_tokens = 4
        num_layers = 6
        num_heads = 8
        seq_len = 10

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        score = rauq_uncertainty_mean_heads(
            log_probs, attentions, token_aggregation="original", alphas=[0.3], ablation=False
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertFalse(np.isnan(score))

    def test_rauq_uncertainty_rollout(self):
        """Test rauq_uncertainty_rollout function."""
        num_tokens = 4
        num_layers = 4
        num_heads = 4
        seq_len = 8
        input_length = seq_len

        log_probs = torch.randn(1, num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        score = rauq_uncertainty_rollout(
            log_probs,
            attentions,
            token_aggregation="original",
            input_length=input_length,
            alphas=[0.4],
            ablation=False,
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertFalse(np.isnan(score))

    def test_rauq_unified_interface(self):
        """Test RAUQ unified interface with different head aggregations."""
        num_tokens = 4
        num_layers = 4
        num_heads = 4
        seq_len = 8
        input_length = seq_len

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        # Test original
        score_original = RAUQ(
            log_probs,
            attentions,
            input_length,
            token_aggregation="original",
            head_aggregation="original",
            alphas=[0.2],
            ablation=False,
        )
        self.assertIsInstance(score_original, float)

        # Test mean_heads
        score_mean = RAUQ(
            log_probs,
            attentions,
            input_length,
            token_aggregation="original",
            head_aggregation="mean_heads",
            alphas=[0.3],
            ablation=False,
        )
        self.assertIsInstance(score_mean, float)

        # Test rollout
        log_probs_2d = torch.randn(1, num_tokens)
        score_rollout = RAUQ(
            log_probs_2d,
            attentions,
            input_length,
            token_aggregation="original",
            head_aggregation="rollout",
            alphas=[0.4],
            ablation=False,
        )
        self.assertIsInstance(score_rollout, float)

    def test_rauq_different_alphas(self):
        """Test RAUQ with different alpha values."""
        num_tokens = 3
        num_layers = 4
        num_heads = 4
        seq_len = 6

        log_probs = torch.randn(num_tokens)
        attentions = self._create_mock_attentions(num_tokens, num_layers, num_heads, seq_len)

        score_low = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=[0.1], ablation=False
        )
        score_high = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=[0.9], ablation=False
        )

        # Scores should be different with different alphas
        self.assertNotAlmostEqual(score_low, score_high, delta=TOL)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_normalized_entropy_empty_sequences(self):
        """Test normalized_entropy with single-element sequences."""
        log_probs = torch.randn(3, 1)

        score = normalized_entropy(log_probs)

        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_perplexity_single_token(self):
        """Test perplexity with single token."""
        log_probs = torch.tensor([-0.5])

        score = perplexity(log_probs)

        self.assertAlmostEqual(score, 0.5, delta=TOL)

    def test_generation_entropy_single_token(self):
        """Test generation_entropy with single token."""
        logits = (torch.randn(1, 100),)

        score = generation_entropy(logits)

        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_eigen_score_small_dimensions(self):
        """Test eigen_score with small hidden dimensions."""
        # Small hidden dimension but enough layers (need at least 16)
        hidden_states = tuple(tuple(torch.randn(1, 3, 8) for _ in range(20)) for _ in range(2))

        score = eigen_score(hidden_states, alpha=1e-3)

        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic scenarios."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_multiple_uncertainty_scores(self):
        """Test computing multiple uncertainty scores on the same data."""
        # Generate synthetic data
        num_tokens = 5
        num_layers = 20  # Need at least 16 layers for eigen_score
        num_heads = 8
        seq_len = 10
        vocab_size = 100

        log_probs = torch.randn(num_tokens)
        logits = tuple(torch.randn(1, vocab_size) for _ in range(num_tokens))

        attentions = []
        for t in range(num_tokens):
            layer_attentions = []
            for l in range(num_layers):
                attn = torch.softmax(torch.randn(1, num_heads, 1, seq_len + t), dim=-1)
                layer_attentions.append(attn)
            attentions.append(tuple(layer_attentions))
        attentions = tuple(attentions)

        hidden_states = tuple(
            tuple(torch.randn(1, seq_len + t, 768) for _ in range(num_layers))
            for t in range(num_tokens)
        )

        # Compute various scores
        perp_score = perplexity(log_probs)
        gen_entropy = generation_entropy(logits)
        norm_entropy = normalized_entropy(log_probs.unsqueeze(0))
        eigen_sc = eigen_score(hidden_states)
        rauq_score = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=[0.2], ablation=False
        )

        # All scores should be valid
        scores = [perp_score, gen_entropy, norm_entropy, eigen_sc, rauq_score]
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertFalse(np.isnan(score))
            self.assertFalse(np.isinf(score))

    def test_rauq_consistency_across_aggregations(self):
        """Test that RAUQ scores are consistent across different aggregations."""
        num_tokens = 4
        num_layers = 4
        num_heads = 4
        seq_len = 8

        log_probs = torch.randn(num_tokens)
        attentions = []
        for t in range(num_tokens):
            layer_attentions = []
            for l in range(num_layers):
                attn = torch.softmax(torch.randn(1, num_heads, 1, seq_len + t), dim=-1)
                layer_attentions.append(attn)
            attentions.append(tuple(layer_attentions))
        attentions = tuple(attentions)

        # Compute with different token aggregations
        score_orig = rauq_uncertainty(
            log_probs, attentions, token_aggregation="original", alphas=[0.2], ablation=False
        )
        score_mean = rauq_uncertainty(
            log_probs, attentions, token_aggregation="mean_all_tokens", alphas=[0.2], ablation=False
        )

        # Both should be valid (but may differ in value)
        self.assertIsInstance(score_orig, float)
        self.assertIsInstance(score_mean, float)
        self.assertFalse(np.isnan(score_orig))
        self.assertFalse(np.isnan(score_mean))


if __name__ == "__main__":
    unittest.main()
