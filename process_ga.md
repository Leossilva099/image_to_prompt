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

4 - Population convergence

After ~14 iterations, all MUT-0 (crossover-only) candidates became nearly identical — minor word swaps around the same sentence structure ("A radiant close-up of freshly squeezed orange juice in a frosted glass..."). Crossover of similar parents produces similar children, which then dominate the population, leaving no diversity to exploit.
Roulette wheel ineffective with compressed fitness -All candidates stabilised between fitness 0.86–0.87. With such small differences, fitness-proportionate selection is essentially uniform random selection ,there is no meaningful selection pressure. The best candidate from the OPRO seed (0.8779) was never beaten across all 20 GA iterations.
Mutation insufficiently counteracts convergence - At 30% mutation rate, many iterations had zero or one mutation. When mutation did fire, the resulting prompts (dramatic lighting, dark backgrounds, industrial surfaces) scored lower and were consistently eliminated from the population. This reinforced convergence rather than fighting it.

5 - Fixes applied

- Rank-based selection: replaced roulette wheel. Each candidate is ranked by fitness and assigned a weight equal to its rank (1 to 20). The best candidate is 20x more likely to be selected than the worst, regardless of how close the absolute fitness values are.
- Crossover prompt: added explicit instruction to use a different sentence structure and opening words than both parents, to prevent the crossover from producing near-identical children.
- Progressive mutation rate: starts at 0.3 and increases by 0.05 each iteration, capped at 0.9. By iteration 10 the rate is 0.75, forcing more exploration as the run progresses.

6 - Results after fixes (20 iterations)

Best fitness never improved — stuck at 0.8779 (the OPRO seed) across all 20 iterations. Mean crept from 0.8690 to 0.8709, only because bad mutations are eliminated from the population.

The progressive mutation rate made exploration worse, not better. The LLM consistently mutates toward "dramatic/industrial/dark/moody" prompts — harsh spotlights, metallic surfaces, black backgrounds — which are exactly what the fitness function punishes. The target image has warm soft light on a wooden surface, and any deviation from that is penalized. Mutation rate 0.90 by iteration 13 meant almost every candidate was a mutation, and almost every mutation scored 0.60–0.82 vs the crossover candidates scoring 0.86+.

The core problem: the GA was seeded from a converged OPRO population that had already found the optimal region. The LLM's mutation bias (toward dramatic/edgy aesthetics) consistently explores away from what the fitness function rewards. Increasing mutation rate amplified this — more mutations meant more bad candidates and no improvement.
