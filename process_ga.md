# GA Process Notes

The initial mutation prompt instructed the LLM to rewrite the prompt "drastically", allowing it to change the subject, setting, style, and atmosphere freely. The only constraint was to keep "the same general scene type", which proved too vague.
In practice the model ignored the subject entirely, a prompt about orange juice on a wooden surface was mutated into a desert landscape at sunset. This defeats the purpose of mutation in this context: we want exploration around the same image, not random scene generation.

The mutation prompt was tightened to explicitly forbid changing the main subject. The LLM is now free to vary lighting, artistic style, and mood, but the subject must remain identical. Temperature was also lowered from 1.1 to 1.0 to reduce erratic outputs.
