import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

DIMENSIONS = ["subject", "lighting", "style", "composition", "mood"]


def load_llm(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    llm.eval()
    return llm, processor


def unload_llm(llm, processor):
    del llm, processor
    torch.cuda.empty_cache()


def _format_top_k(candidates, top_k=5):
    top = sorted(candidates, key=lambda x: x["fitness"], reverse=True)[:top_k]
    lines = [f"Score {c['fitness']:.3f}: {c['prompt'].strip()}" for c in top]
    return "\n".join(lines)


def _generate_one_dimension(image, candidates, llm, processor, dimension, temperature):
    top_refs = _format_top_k(candidates, top_k=5)

    DIMENSION_GUIDE = {
        "subject": (
            "subject — its appearance, material, colour, and details",
            "Start with the subject noun phrase'",
        ),
        "lighting": (
            "lighting — quality, direction, colour temperature, and shadows",
            "Start with the lighting condition'",
        ),
        "style": (
            "artistic style and rendering quality",
            "START with the artistic style and rendering quality: medium, technique, and visual treatment.",
        ),
        "composition": (
            "framing, perspective, depth of field, and spatial arrangement",
            "Start with a framing descriptor'",
        ),
        "mood": (
            "emotional tone and narrative atmosphere",
            "Start with a mood clause'",
        ),
    }

    emphasis, opening_rule = DIMENSION_GUIDE[dimension]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive the target image and the top-scoring prompts found so far.\n"
                "Your task: write ONE new, complete prompt that reproduces the target image "
                f"and particularly excels at **{emphasis}**.\n\n"
                f"OPENING RULE: {opening_rule}. "
                "Your prompt MUST begin exactly as the rule prescribes - use different wording from any reference.\n\n"
                "Rules:\n"
                " - Cover all visual aspects (subject, lighting, style, composition, mood).\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence**.\n"
                " - Do NOT copy any reference's opening words. Recombine and improve.\n"
                " - Output ONLY the prompt text, nothing else."
            ),

        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"TOP REFERENCES:\n{top_refs}\n\n"
                        f"Write a complete prompt that starts as the opening rule prescribes and excels at **{emphasis}**. "
                        "Maximum 70 tokens. End with a complete sentence. **NEVER cut mid-sentence**. Output ONLY the prompt text."
                    ),
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm.device)

    with torch.no_grad():
        generated_ids = llm.generate(
            **inputs,
            max_new_tokens=72,
            temperature=temperature,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_initial_candidates(
    image_path,
    llm,
    processor,
    candidates,
    n_candidates=5,
    temperature=0.9,
):
    image = Image.open(image_path).convert("RGB")
    results = []
    for i in range(n_candidates):
        dimension = DIMENSIONS[i % len(DIMENSIONS)]
        prompt = _generate_one_dimension(image, candidates, llm, processor, dimension, temperature)
        results.append(prompt)
        print(f"  [{i+1:02d}/{n_candidates}] {prompt}")
    print(f" {len(results)} generated candidates")
    return results
