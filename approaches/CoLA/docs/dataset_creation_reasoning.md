## Dataset Creation Reasoning

Since we are unsure about how to sample and select the data and how much and which langauges this doc should bring some order to the process

### Decisions to be done
1. Which langauges to inclde
2. How much of each langauge to include


### Which langauges to include
To decide which langs, we first define requirements for this questions.

We want a dataset which is:
1. Diverse, board linguistic features. Since we analyze how good can the approach capture multilingiual diverse data
2. evaluable. We want to explicityl evaluate as much as we can. So we aim to select and prefern lagns we can evaluate
2.1 The benchmarks with the most langs are (to my knowledge): flores and belebele (200 and 122 langs)
3. available in fineweb. Fineweb is to our knowledge the largest multilingual dataset
4. CoLA-expert optimal / optiomal for our novel approach. We need clusters of languages (e.g. 4 Romance, 4 Germanic) so shared A matrices can learn family patterns.

### Decision on Language Grouping
To satisfy requirement #4 ("CoLA-expert optimal"), we will use **Data-Driven Grouping (LLM Embeddings)**.
- **Why**: Linguistic families (e.g., "Indo-European") are often too broad or unbalanced for parameter sharing. A data-driven approach using the model's own embeddings ensures that languages sharing parameters are actually similar in the model's representation space.
- **Method**: Stratified Cluster Sampling on `llm_embeddings.csv` (Llama-3.2-1B).
  1. Cluster languages into $K$ groups using K-Means.
  2. From each cluster, select the best representative (closest to center) for **High**, **Medium**, and **Low** resource tiers.
- **Outcome**: A diverse set of languages that covers the semantic space while ensuring representation across resource levels.
