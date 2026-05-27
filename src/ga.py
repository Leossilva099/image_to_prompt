import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DIVERSITY_THRESHOLD = 0.85


def load_llm(model_id=MODEL_ID):
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm.eval()
    return llm, tokenizer


def unload_llm(llm, tokenizer):
    del llm, tokenizer
    torch.cuda.empty_cache()


def cosine_sim_to_pool(prompt, pool_prompts, clip_model, clip_processor):
    if not pool_prompts:
        return 0.0
    texts = pool_prompts + [prompt]
    inputs = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(clip_model.device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    features = torch.nn.functional.normalize(features, dim=-1)
    sims = (features[-1:] @ features[:-1].T).squeeze(0)
    return float(sims.max().item())


def is_diverse_enough(prompt, population, clip_model, clip_processor, threshold=DIVERSITY_THRESHOLD):
    return cosine_sim_to_pool(prompt, [c["prompt"] for c in population], clip_model, clip_processor) < threshold


def _chat(llm, tokenizer, messages, temperature=0.9, max_new_tokens=72):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    with torch.no_grad():
        generated_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _rank_select(population):
    ranked = sorted(population, key=lambda x: x["fitness"])
    n = len(ranked)
    weights = list(range(1, n + 1))
    total = sum(weights)
    pick = random.uniform(0, total)
    cumulative = 0.0
    for candidate, w in zip(ranked, weights):
        cumulative += w
        if cumulative >= pick:
            return candidate
    return ranked[-1]


def _crossover(llm, tokenizer, parent_a, parent_b, temperature=0.9):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive two parent prompts. Combine their strongest visual elements "
                "into one new prompt.\n\n"
                "Rules:\n"
                " - Cover all visual aspects (subject, lighting, style, composition, mood).\n"
                " - Use a different sentence structure and opening words than both parents.\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence.**\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Parent A (fitness {parent_a['fitness']:.3f}):\n{parent_a['prompt'].strip()}\n\n"
                f"Parent B (fitness {parent_b['fitness']:.3f}):\n{parent_b['prompt'].strip()}\n\n"
                "Combine the best elements. Different structure and opening than both parents. "
                "Maximum 70 tokens. End with a complete sentence. Output ONLY the prompt text."
            ),
        },
    ]
    return _chat(llm, tokenizer, messages, temperature=temperature)


def _mutate(llm, tokenizer, prompt, temperature=0.9):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive a prompt. Change exactly ONE adjective or short descriptor "
                "(a lighting word, texture word, colour word, or mood word) to a different but plausible alternative. "
                "Everything else must remain word-for-word identical.\n\n"
                "Rules:\n"
                " - Change only ONE word or short phrase.\n"
                " - Do NOT change the subject, objects, or overall scene.\n"
                " - Output ONLY the modified prompt, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt:\n{prompt.strip()}\n\n"
                "Change one adjective or descriptor. Output ONLY the modified prompt."
            ),
        },
    ]
    return _chat(llm, tokenizer, messages, temperature=temperature)


def evolve(llm, tokenizer, population, clip_model, clip_processor,
           n_candidates=5, iteration=1, base_mutation_rate=0.3, temperature=0.9):
    new_prompts = []
    for i in range(n_candidates):
        parent_a = _rank_select(population)
        parent_b = _rank_select(population)
        retries = 0
        while parent_b["prompt"] == parent_a["prompt"] and len(population) > 1 and retries < 5:
            parent_b = _rank_select(population)
            retries += 1

        child = _crossover(llm, tokenizer, parent_a, parent_b, temperature)
        flag = 0
        if random.random() < base_mutation_rate:
            flag = 1
            child = _mutate(llm, tokenizer, child)

        pool = [c["prompt"] for c in population] + new_prompts
        if cosine_sim_to_pool(child, pool, clip_model, clip_processor) >= DIVERSITY_THRESHOLD:
            print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] skipped (too similar)")
            continue

        new_prompts.append(child)
        print(f"MUT-{flag}-[{i+1:02d}/{n_candidates}] {child}")

    print(f" {len(new_prompts)} evolved candidates")
    return new_prompts
