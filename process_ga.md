# GA Process Notes

1- The initial mutation prompt instructed the LLM to rewrite the prompt "drastically", allowing it to change the subject, setting, style, and atmosphere freely. The only constraint was to keep "the same general scene type", which proved too vague.
In practice the model ignored the subject entirely, a prompt about orange juice on a wooden surface was mutated into a desert landscape at sunset. This defeats the purpose of mutation in this context: we want exploration around the same image, not random scene generation.

2 - The mutation prompt was tightened to explicitly forbid changing the main subject. The LLM is now free to vary lighting, artistic style, and mood, but the subject must remain identical. Temperature was also lowered from 1.1 to 1.0 to reduce erratic outputs.

3- Again the prompt 

```
role": "system","content": ("You are an expert prompt engineer for LCM diffusion models.\n""You will receive a prompt. Rewrite it by freely changing lighting, artistic style, ""composition, mood, and atmosphere — but the main subject must remain exactly the same.\n\n""Rules:\n"" - NEVER change the main subject.\n"" - Maximum 70 tokens. Every sentence must be complete. NEVER cut mid-sentence.\n"" - Output ONLY the new prompt text, nothing else."    ),
```

produced (w/ mutation)  more related prompts to what we wanted but we start to get childs like "A dramatic macro shot of steaming hot chocolate in a rustic mug" in the pic of orange juice

changed to a more restritive one , with rules 

```
ABSOLUTE RULES:\n"" - Every object, food, drink, and prop in the original must appear in your output.\n"" - Do NOT introduce any new objects, foods, or drinks not present in the original.\n"" - Maximum 70 tokens. Every sentence must be complete. NEVER cut mid-sentence.\n"" - Output ONLY the new prompt text, nothing else
```
