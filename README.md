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

##### Dataset - Enailed Polarity

###### Case 1:
- Prefix: "Given a fact, answer the following question with a yes or a no."
- Suffix: "What is the entailed polarity?"
- Max token: 1
- Temparature: 0

##### Dataset - Analytical Entailment

- Max token: 1
- Temparature: 0

###### Case 1:
- Prefix: "Determine whether the following pairs of sentences embody an entailment relation or not."
- Suffix: "Is this entailment?"

###### Case 2:
- Prefix: "Determine whether the following pairs of sentences embody an entailment relation or not."
- Suffix: "Is this entailment Yes/No?"

##### Dataset - Super GLUE RTE

- Max token: 1
- Temparature: 0

###### Case 1:
- Prefix: "Determine whether the following pairs of sentences embody an entailment relation or not."
- Suffix: "Is this entailment?"

###### Case 2:
- Prefix: "Determine whether the following pairs of sentences embody an entailment relation or not."
- Suffix: "Is this entailment Yes/No?"

### Data Pre-processing

- Step 1: Converting Data into a common format

- Step 2: Removing any data record that is filtered out at the Azure OpenAI

Link to the pre-processed dataset used - [Pre-processed Dataset](./Dataset/Preprocessed)
### Execution

- Step 3: Capturing the responses for each model for each dataset

### Data Post-processing

- Step 4: Trimming the responses and converting to lower case (since the max token is kept at 1)

### Results

- Enailed Polarity:
  - Case 1:
- Analytical Entailment:
  - Case 1:
  - Case 2:
- Super GLUE RTE:
  - Case 1:
  - Case 2:

### Analysis

##### Cost v/s F1-score

##### Prompt type v/s F1-score

##### Cost v/s Prompt type
