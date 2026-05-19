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
        "subject":     "make the main subject, its appearance, details, and physical description, the defining quality. Also cover lighting, style, and composition.",
        "lighting":    "make the lighting, shadows, and atmosphere the defining quality. Also cover the subject, setting, and style.",
        "style":       "make the artistic style and rendering quality (e.g. cinematic, photorealistic, painterly) the defining quality. Also cover subject and lighting.",
        "composition": "make the framing, perspective, depth of field, and spatial arrangement the defining quality. Also cover subject and style.",
        "mood":        "make the emotional tone and narrative feeling the defining quality. Also cover subject, lighting, and style.",
    }

    guide = DIMENSION_GUIDE[dimension]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive the top-scoring prompts found so far.\n"
                "Your task: write ONE new, complete prompt that synthesises the best visual elements "
                "from those references and particularly excels at "
                f"**{dimension}**.\n\n"
                f"EMPHASIS: {guide}\n\n"
                "Rules:\n"
                " - Write a complete prompt covering all visual aspects (subject, lighting, style, composition, mood).\n"
                " - Maximum 70 tokens. Every sentence must be complete."
                " **VERY IMPORTANT** - NEVER cut mid-sentence.\n"
                " - Do NOT copy any prompt word for word, recombine and improve.\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TOP REFERENCES:\n{top_refs}\n\n"
                f"Write a complete prompt that synthesises the best elements above and particularly excels at **{dimension}**. "
                f"Maximum 70 tokens. End with a complete sentence. **NEVER cut mid-sentence**. Output ONLY the prompt text."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=72,
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
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")
    print(f" {len(results)} generated candidates")
    return results
