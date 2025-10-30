from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from runia.llm_uncertainty.scores import compute_uncertainties

PROMPT = "What is the capital of France?"


def main():
    # Load model

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token = "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")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",  token = "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")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # Alternative HF token: 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

    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )

    uncertainty_requests = [
        {"method_name": "eigen_score"},
        {"method_name": "normalized_entropy"},
        {"method_name": "semantic_entropy"},
        {"method_name": "perplexity"},
        {"method_name": "generation_entropy"},
        {
            "method_name": "RAUQ",
            "token_aggregation": "original",  # or 'original'
            "head_aggregation": "mean_heads",  # or 'original' or 'mean_heads'
            "alphas": [0.2, 0.4, 0.6],  # for ablation study
            "ablation": True,  # to get scores for all alphas
        },
        {
            "method_name": "RAUQ",
            "token_aggregation": "original",  # or 'original'
            "head_aggregation": "rollout",  # or 'original' or 'mean_heads'
            "ablation": False,
        },
    ]

    generated_text, scores = compute_uncertainties(
        model,
        tokenizer,
        PROMPT,
        uncertainty_requests,
        gen_config,
        num_samples=10,
    )

    print("Generated Text:", generated_text)
    print("Uncertainty Scores:", scores)


if __name__ == "__main__":
    main()
