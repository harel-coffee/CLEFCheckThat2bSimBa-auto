# FEATURE CORRELATION

| Spearman Correlation       |sbert|infersent|universal|sim_cse|sequence_matcher|levenshtein| jacc_char| jacc_tok|ne   |main_syns|words|subjects|score|
|----------------------------|-----|---------|---------|-------|----------------|-----------|----------|---------|-----|---------|-----|--------|-----|
|sbert                       |1.0  |0.02     |0.24     |0.12   |0.02            |-0.0       |0.03      |0.03     |0.04 |0.03     |0.05 |0.01    |0.02 |
|infersent                   |0.02 |1.0      |0.09     |0.02   |-0.02           |-0.1       |0.07      |0.08     |0.02 |0.1      |0.1  |0.02    |0.01 |
|universal                   |0.24 |0.09     |1.0      |0.12   |0.06            |0.02       |0.04      |0.06     |0.1  |0.06     |0.08 |0.01    |0.03 |
|sim_cse                     |0.12|0.02|0.12|1.0|0.15|0.1|0.11|0.12|0.22|0.14|0.16|-0.01|0.1|
|sequence_matcher            |0.02|-0.02|0.06|0.15|1.0|0.55|0.07|0.31|0.12|0.03|0.08|0.04|0.07|
|levenshtein|-0.0|-0.1|0.02|0.1|0.55|1.0|-0.21|0.05|0.01|-0.28|-0.25|-0.05|0.05|
|jacc_char|0.03|0.07|0.04|0.11|0.07|-0.21|1.0|0.29|0.13|0.29|0.32|0.11|0.05|
|jacc_tok|0.03|0.08|0.06|0.12|0.31|0.05|0.29|1.0|0.21|0.41|0.62|0.24|0.09|
|ne|0.04|0.02|0.1|0.22|0.12|0.01|0.13|0.21|1.0|0.17|0.32|0.03|0.09|
|main_syns|0.03|0.1|0.06|0.14|0.03|-0.28|0.29|0.41|0.17|1.0|0.67|0.09|0.08|
|words|0.05|0.1|0.08|0.16|0.08|-0.25|0.32|0.62|0.32|0.67|1.0|0.08|0.09|
|subjects|0.01|0.02|0.01|-0.01|0.04|-0.05|0.11|0.24|0.03|0.09|0.08|1.0|0.05|
|score|0.02|0.01|0.03|0.1|0.07|0.05|0.05|0.09|0.09|0.08|0.09|0.05|1.0|


## Top Features

1. Sim CSE
2. Jacc_tokens, Named Entities, Words
3. Synonyms
4. Sequence Matcher   
5. Levenshtein, Jacc_chars, Subjects
6. Universal Sentence Encoder
6. Sentence Bert
7. Infersent

## Feature Correlation between each other

1. Sim CSE - Named Entities
2. Jacc_tok- Words - Main Syns
3. Jacc_tok - Sequence Matcher
4. Jacc_tok - Jacc_chars
5. Jacc_tok -Named Entities
6. Named Entities - Words
7. Words - Jacc Chars
8. Sequence Matcher- Levenshtein
9. Jacc-chars - Words
10. Subjects - Jacc_tokens
11. Universal Sentence Encoder - Sbert
12. Infersent

# Similarity Scores

1. Top Sim CSE Scores
2. Top Jaccard Tokens
3. Top Named Entity Similarity
4. Top Words Similarity
5. Top Synonym Similarity

# Best scores so far
UnsupervisedFeatureSetGenerator(['universal', 'sim_cse'], 'pp1'): 0.4135

binary classification: 0.4335






