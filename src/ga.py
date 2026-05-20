import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_llm(model_id=MODEL_ID):
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm.eval()
    return llm, tokenizer


def unload_llm(llm, tokenizer):
    del llm, tokenizer
    torch.cuda.empty_cache()


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


def _roulette_select(population):
    fitnesses = [max(c["fitness"], 0.0) for c in population]
    total = sum(fitnesses)
    if total == 0:
        return random.choice(population)
    pick = random.uniform(0, total)
    cumulative = 0.0
    for candidate, f in zip(population, fitnesses):
        cumulative += f
        if cumulative >= pick:
            return candidate
    return population[-1]


def _crossover(llm, tokenizer, parent_a, parent_b, temperature=0.9):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive two parent prompts with their fitness scores.\n"
                "Your task: write ONE new, complete prompt that combines the strongest visual "
                "elements from both parents.\n\n"
                "Rules:\n"
                " - Cover all visual aspects (subject, lighting, style, composition, mood).\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence.**\n"
                " - Do NOT copy either prompt word for word — recombine and improve.\n"
                " - Output ONLY the prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Parent A (fitness {parent_a['fitness']:.3f}):\n{parent_a['prompt'].strip()}\n\n"
                f"Parent B (fitness {parent_b['fitness']:.3f}):\n{parent_b['prompt'].strip()}\n\n"
                "Combine the best elements into one prompt. "
                "Maximum 70 tokens. End with a complete sentence. Output ONLY the prompt text."
            ),
        },
    ]
    return _chat(llm, tokenizer, messages, temperature=temperature)


def _mutate(llm, tokenizer, prompt, temperature=1.0):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for LCM diffusion models.\n"
                "You will receive a prompt. Rewrite it by freely changing lighting, artistic style, "
                "composition, mood, and atmosphere — but the main subject must remain exactly the same.\n\n"
                "Rules:\n"
                " - NEVER change the main subject.\n"
                " - Maximum 70 tokens. Every sentence must be complete. **NEVER cut mid-sentence.**\n"
                " - Output ONLY the new prompt text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt to mutate:\n{prompt.strip()}\n\n"
                "Keep the subject identical. Freely change lighting, style, composition, or mood. "
                "Maximum 70 tokens. End with a complete sentence. Output ONLY the prompt text."
            ),
        },
    ]
    return _chat(llm, tokenizer, messages, temperature=temperature)


def evolve(llm, tokenizer, population, n_candidates=5, mutation_rate=0.3, temperature=0.9):
    """
    One GA generation over a population of dicts with keys: prompt, fitness, clip, lpips, rmse.
    Returns a list of new prompt strings (length == n_candidates).

    Steps per child:
      1. Roulette-wheel selection (fitness-proportionate) to pick two parents.
      2. LLM crossover.
      3. Destructive LLM mutation with probability mutation_rate.
    """
    new_prompts = []
    for i in range(n_candidates):
        parent_a = _roulette_select(population)
        parent_b = _roulette_select(population)
        retries = 0
        while parent_b["prompt"] == parent_a["prompt"] and len(population) > 1 and retries < 5:
            parent_b = _roulette_select(population)
            retries += 1

        child = _crossover(llm, tokenizer, parent_a, parent_b, temperature)
        flag = 0
        if random.random() < mutation_rate:
            flag = 1
            child = _mutate(llm, tokenizer, child)

        new_prompts.append(child)
        print(f"{flag}-[{i+1:02d}/{n_candidates}] {child}")

    print(f" {len(new_prompts)} evolved candidates")
    return new_prompts
