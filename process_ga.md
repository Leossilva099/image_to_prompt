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

7 - Fixes applied (second round)

Three changes were made simultaneously to address the convergence and mutation bias problems identified in section 6.

Conservative mutation: The mutation prompt was rewritten to change exactly ONE adjective or short descriptor (lighting, texture, colour, or mood word), with everything else remaining word-for-word identical. This directly counters the LLM's bias toward dramatic rewrites — the model can no longer substitute the scene or swap objects, only nudge one descriptor at a time. Temperature kept at 0.9. Progressive mutation rate was removed; rate is fixed at 0.3.

CLIP diversity filter: Before a new candidate is accepted into the generation, its CLIP text embedding is compared against all current population members plus already-accepted new candidates. Any candidate with cosine similarity ≥ 0.85 to the pool is discarded. This prevents the population from collapsing into near-duplicate prompts even when crossover produces structurally similar children.

Rank-based selection (combined with the above): Already introduced in section 5, but now running together with the two fixes above. Each candidate is ranked by fitness and assigned a weight equal to its rank (1 to 20), giving the best candidate 20× more selection pressure than the worst regardless of absolute fitness differences.

8 - Results after second round of fixes (20 iterations)

OPRO seed best: 0.8489. GA best: 0.8659 (iteration 4). For the first time, the GA improved over its seed (+0.017). Mean rose slowly from 0.8371 to 0.8443 across 20 iterations.

Best prompt found (fitness 0.8659):
"A smooth, dark surface supports a glass filled with frothy orange juice, garnished with a vibrant orange slice and a crystallized sugar cube, surrounded by scattered orange segments and zest, bathed in soft, golden natural light that highlights the inviting, juicy colors and textures of the scene."

New problem: CLIP diversity filter too aggressive. From iteration 5 onward, the filter rejected most candidates — iterations 5, 6, and 7 produced only 1 candidate each, and iteration 8 produced 0. The best was found at iteration 4 and never improved across the remaining 16 iterations. With a converged population, even crossover of different parents produces children that score ≥ 0.85 cosine similarity to the pool, so the filter blocks exploration instead of helping it.

Root cause: threshold 0.85 is too tight once the population has converged. The filter is working as intended (preventing exact duplicates) but is calibrated for a diverse population. With 20 candidates already clustered in the same semantic region, nearly any new child lands inside the rejection zone.
