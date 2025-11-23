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

**Opportunity & Decision:**
Intersecting these requirements gives us a pool of **108 evaluatable languages**.
From this pool, we will select **22 languages** using clustering to ensure we have both diversity and family clusters (CoLA-optimal).
