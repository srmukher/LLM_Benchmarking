# LLM_Benchmarking
Benchmarking of LLM models

## Datasets
- Entailed Polarity Dataset: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/entailed_polarity/task.json
- Analytical Entailment Dataset: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/analytic_entailment/task.json

Link to the original dataset used - [Original Dataset](./Dataset/Original)

### Assumptions/Parameters

- Type of Benchmarking: Entailment recognition in Yes/No.
- Models: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#gpt-3-models
  - text-davinci-003
  - text-curie-001
  - text-babbage-001
  - text-ada-001
  - gpt-35-turbo

### Prompts:
1. Prompt1: "{premise}. {hypothesis} yes or no?",  # no instruction  with response options (Yes/No)
2. Prompt2: "{prefix}. {premise}. {hypothesis}.",  # instruction with only prefix
3. Prompt3: "{premise}. {intermediate}. {hypothesis}.",  # instruction with only intermediate instruction
4. Prompt4: "{premise}. {hypothesis}? {suffix}.",  # instruction with only suffix

- prefix = "Given the fact, answer the following question with **yes/no**."
- suffix = "Given the previous fact, answer the following question with **yes/no**."
- intermediate = "Given the previous fact, answer the following question with **yes/no**."
