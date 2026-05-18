import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DIMENSIONS = ["subject", "lighting", "style", "composition", "mood"]


def load_llm(model_id="Qwen/Qwen2.5-7B-Instruct"):
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm.eval()
    return llm, tokenizer


def _format_top_k(candidates, top_k=5):
    top = sorted(candidates, key=lambda x: x["fitness"], reverse=True)[:top_k]
    lines = []
    for c in top:
        lines.append(f"Score {c['fitness']:.3f}: {c['prompt'].strip()}")
    return "\n".join(lines)


def _generate_one_dimension(candidates, llm, tokenizer, dimension, temperature):
    top_refs = _format_top_k(candidates, top_k=5)

    DIMENSION_GUIDE = {
        "subject":     "the main subject: its appearance, details, and physical description. Do NOT describe lighting, style tags, or composition.",
        "lighting":    "lighting, shadows, and atmosphere only. Do NOT describe the subject in detail or add style/quality tags.",
        "style":       "artistic style, rendering quality, and visual technique only (e.g. painterly, cinematic, photorealistic). Do NOT describe subject or lighting.",
        "composition": "framing, perspective, depth of field, and spatial arrangement only. Do NOT describe colors, mood, or style.",
        "mood":        "emotional tone, narrative, and mood only. Do NOT describe physical details, lighting specifics, or technical tags.",
    }

    guide = DIMENSION_GUIDE[dimension]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive the top-scoring prompts found so far.\n"
                "Your task: write ONE new prompt that keeps the best visual elements "
                "from those references but is restructured to focus EXCLUSIVELY on "
                f"**{dimension}**.\n\n"
                f"FOCUS ONLY ON: {guide}\n\n"
                "Rules:\n"
                "• Under 70 tokens, complete sentence.\n"
                "• Do NOT copy any prompt verbatim — recombine and improve.\n"
                "• Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TOP REFERENCES:\n{top_refs}\n\n"
                f"Write a new prompt focused EXCLUSIVELY on **{dimension}**. "
                f"Under 70 tokens. Output ONLY the prompt text."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=70,
        temperature=temperature,
        do_sample=True,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_initial_candidates(
    llm,
    tokenizer,
    candidates,
    n_candidates=5,
    temperature=0.9,
):
    results = []
    for i in range(n_candidates):
        dimension = DIMENSIONS[i % len(DIMENSIONS)]
        prompt = _generate_one_dimension(candidates, llm, tokenizer, dimension, temperature)
        results.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] [{dimension}] {prompt}")
    print(f" {len(results)} generated candidates")
    return results
