# LLM_Benchmarking
Benchmarking of LLM models

## Datasets
- Entailed Polarity Dataset: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/entailed_polarity/task.json
- Analytical Entailment Dataset: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/analytic_entailment/task.json
- Super GLUE Recognizing Textual Entailment (RTE): https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
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
1. Prompt1: "{premise}. {hypothesis}?",  # no instruction
2. Prompt2: "{premise}. {hypothesis} yes or no?",  # no instruction  with response options (Yes/No)
3. Prompt3: "{prefix}. {premise}. {hypothesis}.",  # instruction with only prefix
4. Prompt4: "{premise}. {intermediate}. {hypothesis}.",  # instruction with only intermediate instruction
5. Prompt5: "{premise}. {hypothesis}? {suffix}.",  # instruction with only suffix

- prefix = "Given the fact, answer the following question with a yes or no."
- suffix = "Given the previous fact, answer the following question with a yes or a no."
- intermediate = "Given the previous fact, answer the following question with a yes or a no."
