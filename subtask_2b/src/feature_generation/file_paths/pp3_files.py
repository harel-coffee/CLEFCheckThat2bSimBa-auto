v_claims_directory_pp3 = 'data/politifact-vclaims'
v_claims_df_pp3 = 'data/vclaims_df.pkl'

training_data_pp3 = 'data/pp_speech_data/training_data/pp3_iclaims.queries'
labels_general = 'data/original_twitter_data/training_data/all_train.pkl'

test_data_pp3 = 'data/pp_speech_data/test_data/pp3_queries.tsv'
# Complete Feature Sets

# Training

complete_feature_set_pairs_train_pp3 = 'data/feature_sets/training/pp3/complete_feature_set_pairs_train_pp3.pkl'
complete_feature_set_pairs_train_tsv_pp3 = 'data/feature_sets/training/pp3/complete_feature_set_pairs_train_pp3.tsv'
complete_feature_set_triples_train_pp3 = 'data/feature_sets/training/pp3/complete_feature_set_triples_train_pp3'
complete_feature_set_triples_train_tsv_pp3 = 'data/feature_sets/training/pp3/complete_feature_set_triples_train_pp3.tsv'

# Test

complete_feature_set_pairs_test_pp3 = 'data/feature_sets/test/pp3/complete_feature_set_pairs_test_pp3.pkl'
complete_feature_set_pairs_test_tsv_pp3 = 'data/feature_sets/test/pp3/complete_feature_set_pairs_test_pp3.tsv'
complete_feature_set_triples_test_pp3 = 'data/feature_sets/test/pp3/complete_feature_set_triples_test_pp3'
complete_feature_set_triples_test_tsv_pp3 = 'data/feature_sets/test/pp3/complete_feature_set_triples_test_pp3.tsv'


### Sentence Features ###

# Verified Claims

# Feature 1
sbert_encodings_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/sbert_encodings_vclaims_pp3.pkl'
sbert_encodings_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/sbert_encodings_vclaims_pp3.tsv'
# Feature 2
infersent_encodings_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/infersent_encodings_vclaims_pp3.pkl'
infersent_encodings_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/infersent_encodings_vclaims_pp3.tsv'
# Feature 3
universal_encodings_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/universal_encodings_vclaims_pp3.pkl'
universal_encodings_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/universal_encodings_vclaims_pp3.tsv'
# Feature 4
sim_cse_encodings_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/sim_cse_encodings_vclaims_pp3.pkl'
sim_cse_encodings_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/sim_cse_encodings_vclaims_pp3.tsv'
# Feature 9
ne_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/ne_vclaims_pp3.pkl'
ne_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/ne_vclaims_pp3.tsv'
# Feature 10
main_syms_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/main_syms_tvclaims_pp3.pkl'
main_syms_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/main_syms_vclaims_pp3.tsv'
# Feature 11
words_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/words_vclaims_pp3.pkl'
words_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/words_tvclaims_pp3.tsv'
# Feature 12
subjects_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/subjects_vclaims_pp3.pkl'
subjects_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/subjects_vclaims_pp3.tsv'
# Feature 13
token_number_vclaims_pp3 = 'data/feature_sets/sentence_features/vclaims/pp3/token_number_vclaims_pp3.pkl'
token_number_vclaims_pp3_tsv = 'data/feature_sets/sentence_features/vclaims/pp3/token_number_vclaims_pp3.tsv'

# Training

# Feature 1
sbert_encodings_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/sbert_encodings_train_pp3.pkl'
sbert_encodings_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/sbert_encodings_train_pp3.tsv'
# Feature 2
infersent_encodings_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/infersent_encodings_train_pp3.pkl'
infersent_encodings_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/infersent_encodings_train_pp3.tsv'
# Feature 3
universal_encodings_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/universal_encodings_train_pp3.pkl'
universal_encodings_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/universal_encodings_train_pp3.tsv'
# Feature 4
sim_cse_encodings_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/sim_cse_encodings_train_pp3.pkl'
sim_cse_encodings_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/sim_cse_encodings_train_pp3.tsv'
# Feature 9
ne_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/ne_train_pp3.pkl'
ne_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/ne_train_pp3.tsv'
# Feature 10
main_syms_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/main_syms_train_pp3.pkl'
main_syms_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/main_syms_train_pp3.tsv'
# Feature 11
words_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/words_train_pp3.pkl'
words_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/words_train_pp3.tsv'
# Feature 12
subjects_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/subjects_train_pp3.pkl'
subjects_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/subjects_train_pp3.tsv'
# Feature 13
token_number_training_pp3 = 'data/feature_sets/sentence_features/training/pp3/token_number_train_pp3.pkl'
token_number_training_pp3_tsv = 'data/feature_sets/sentence_features/training/pp3/token_number_train_pp3.tsv'

# Test

# Feature 1
sbert_encodings_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/sbert_encodings_test_pp3.pkl'
sbert_encodings_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/sbert_encodings_test_pp3.tsv'
# Feature 2
infersent_encodings_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/infersent_encodings_test_pp3.pkl'
infersent_encodings_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/infersent_encodings_test_pp3.tsv'
# Feature 3
universal_encodings_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/universal_encodings_test_pp3.pkl'
universal_encodings_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/universal_encodings_test_pp3.tsv'
# Feature 4
sim_cse_encodings_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/sim_cse_encodings_test_pp3.pkl'
sim_cse_encodings_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/sim_cse_encodings_test_pp3.tsv'
# Feature 9
ne_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/ne_test_pp3.pkl'
ne_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/ne_test_pp3.tsv'
# Feature 10
main_syms_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/main_syms_test_pp3.pkl'
main_syms_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/main_syms_test_pp3.tsv'
# Feature 11
words_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/words_test_pp3.pkl'
words_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/words_test_pp3.tsv'
# Feature 12
subjects_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/subjects_test_pp3.pkl'
subjects_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/subjects_test_pp3.tsv'
# Feature 13
token_number_test_pp3 = 'data/feature_sets/sentence_features/test/pp3/token_number_test_pp3.pkl'
token_number_test_pp3_tsv = 'data/feature_sets/sentence_features/test/pp3/token_number_test_pp3.tsv'

### Sentence Similarities ###

# Training

# Feature 1
sbert_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/sbert_sims_train_pp3.pkl'
sbert_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/sbert_sims_train_pp3.tsv'
top_n_sbert_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/top_n_sbert_sims_train_pp3'
top_50_sbert_sims_training_pp3_df = top_n_sbert_sims_training_pp3+'_50.pkl'
top_50_sbert_sims_training_pp3_tsv = top_n_sbert_sims_training_pp3+'_50.tsv'
# Feature 2
infersent_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/infersent_sims_train_pp3.pkl'
infersent_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/infersent_sims_train_pp3.tsv'
top_n_infersent_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/top_n_infersent_sims_train_pp3'
top_50_infersent_sims_training_pp3_df = top_n_infersent_sims_training_pp3+'_50.pkl'
top_50_infersent_sims_training_pp3_tsv = top_n_infersent_sims_training_pp3+'_50.tsv'
# Feature 3
universal_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/universal_sims_train_pp3.pkl'
universal_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/universal_sims_train_pp3.tsv'
top_n_universal_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/top_n_universal_sims_train_pp3'
top_50_universal_sims_training_pp3_df = top_n_universal_sims_training_pp3+'_50.pkl'
top_50_universal_sims_training_pp3_tsv = top_n_universal_sims_training_pp3+'_50.tsv'
# Feature 4
sim_cse_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/sim_cse_sims_train_pp3.pkl'
sim_cse_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/sim_cse_sims_train_pp3.tsv'
top_n_sim_cse_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/top_n_sim_cse_sims_train_pp3'
top_50_sim_cse_sims_training_pp3_df = top_n_sim_cse_sims_training_pp3+'_50.pkl'
top_50_sim_cse_sims_training_pp3_tsv = top_n_sim_cse_sims_training_pp3+'_50.tsv'
# Feature 5
seq_match_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/seq_match_train_pp3.pkl'
seq_match_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/seq_match_train_pp3.tsv'
# Feature 6
levenshtein_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/levenshtein_train_pp3.pkl'
levenshtein_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/levenshtein_train_pp3.tsv'
# Feature 7
jacc_chars_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/jacc_chars_train_pp3.pkl'
jacc_chars_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/jacc_chars_train_pp3.tsv'
# Feature 8
jacc_tokens_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/jacc_tokens_train_pp3.pkl'
jacc_tokens_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/jacc_tokens_train_pp3.tsv'
# Feature 9
ne_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/ne_sims_train_pp3.pkl'
ne_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/ne_sims_train_pp3.tsv'
# Feature 10
main_syms_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/main_syms_sims_train_pp3.pkl'
main_syms_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/main_syms_sims_train_pp3.tsv'
# Feature 11
words_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/words_sims_train_pp3.pkl'
words_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/words_sims_train_pp3.tsv'
# Feature 12
subjects_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/subjects_sims_train_pp3.pkl'
subjects_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/subjects_sims_train_pp3.tsv'
# Feature 13
token_number_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/token_number_sims_training_pp3.pkl'
token_number_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/token_number_sims_training_pp3.tsv'
# Feature 14
ne_ne_ratio_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/ne_ne_ratio_training_pp3.pkl'
ne_ne_ratio_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/ne_ne_ratio_training_pp3.tsv'
# Feature 15
ne_token_ratio_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/ne_token_ratio_training_pp3.pkl'
ne_token_ratio_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/ne_token_ratio_training_pp3.tsv'
# Feature 16
main_syms_token_ratio_sims_training_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/main_syms_token_ratio_training_pp3.pkl'
main_syms_token_ratio_sims_training_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/main_syms_token_ratio_training_pp3.tsv'
# Feature 17
words_token_ratio_sims_pp3 = 'data/feature_sets/sentence_similarities/training/pp3/words_token_ratio_training_pp3.pkl'
words_token_ratio_sims_pp3_tsv = 'data/feature_sets/sentence_similarities/training/pp3/words_token_ratio_training_pp3.tsv'

# Test

# Feature 1
sbert_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/sbert_sims_test_pp3.pkl'
sbert_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/sbert_sims_test_pp3.tsv'
top_n_sbert_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/top_n_sbert_sims_test_pp3'
top_50_sbert_sims_test_pp3_df = top_n_sbert_sims_test_pp3+'_50.pkl'
top_50_sbert_sims_test_pp3_tsv = top_n_sbert_sims_test_pp3+'_50.tsv'
# Feature 2
infersent_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/infersent_sims_test_pp3.pkl'
infersent_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/infersent_sims_test_pp3.tsv'
top_n_infersent_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/top_n_infersent_sims_test_pp3'
top_50_infersent_sims_test_pp3_df = top_n_infersent_sims_test_pp3+'_50.pkl'
top_50_infersent_sims_test_pp3_tsv = top_n_infersent_sims_test_pp3+'_50.tsv'
# Feature 3
universal_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/universal_sims_test_pp3.pkl'
universal_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/universal_sims_test_pp3.tsv'
top_n_universal_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/top_n_universal_sims_test_pp3'
top_50_universal_sims_test_pp3_df = top_n_universal_sims_test_pp3+'_50.pkl'
top_50_universal_sims_test_pp3_tsv = top_n_universal_sims_test_pp3+'_50.tsv'
# Feature 4
sim_cse_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/sim_cse_sims_test_pp3.pkl'
sim_cse_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/sim_cse_sims_test_pp3.tsv'
top_n_sim_cse_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/top_n_sim_cse_sims_test_pp3'
top_50_sim_cse_sims_test_pp3_df = top_n_sim_cse_sims_test_pp3+'_50.pkl'
top_50_sim_cse_sims_test_pp3_tsv = top_n_sim_cse_sims_test_pp3+'_50.tsv'
# Feature 5
seq_match_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/seq_match_test_pp3.pkl'
seq_match_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/seq_match_test_pp3.tsv'
# Feature 6
levenshtein_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/levenshtein_test_pp3.pkl'
levenshtein_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/levenshtein_test_pp3.tsv'
# Feature 7
jacc_chars_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/jacc_chars_test_pp3.pkl'
jacc_chars_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/jacc_chars_test_pp3.tsv'
# Feature 8
jacc_tokens_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/jacc_tokens_test_pp3.pkl'
jacc_tokens_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/jacc_tokens_test_pp3.tsv'
# Feature 9
ne_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/ne_sims_test_pp3.pkl'
ne_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/ne_sims_test_pp3.tsv'
# Feature 10
main_syms_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/main_syms_sims_test_pp3.pkl'
main_syms_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/main_syms_sims_test_pp3.tsv'
# Feature 11
words_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/words_sims_test_pp3.pkl'
words_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/words_sims_test_pp3.tsv'
# Feature 12
subjects_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/subjects_sims_test_pp3.pkl'
subjects_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/subjects_sims_test_pp3.tsv'
# Feature 13
token_number_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/token_number_sims_test_pp3.pkl'
token_number_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/token_number_sims_test_pp3.tsv'
# Feature 14
ne_ne_ratio_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/ne_ne_ratio_test_pp3.pkl'
ne_ne_ratio_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/ne_ne_ratio_test_pp3.tsv'
# Feature 15
ne_token_ratio_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/ne_token_ratio_test_pp3.pkl'
ne_token_ratio_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/ne_token_ratio_test_pp3.tsv'
# Feature 16
main_syms_token_ratio_sims_test_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/main_syms_token_ratio_test_pp3.pkl'
main_syms_token_ratio_sims_test_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/main_syms_token_ratio_test_pp3.tsv'
# Feature 17
words_token_ratio_sims_pp3 = 'data/feature_sets/sentence_similarities/test/pp3/words_token_ratio_test_pp3.pkl'
words_token_ratio_sims_pp3_tsv = 'data/feature_sets/sentence_similarities/test/pp3/words_token_ratio_test_pp3.tsv'

# Combined sentence embedding similarities

## training

train_sbert_infersent_disjunction_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_sbert_infersent_disjunction_pp3.pkl'
train_sbert_infersent_universal_disjunction_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_sbert_infersent_universal_disjunction_pp3.pkl'
train_sbert_infersent_universal_sim_cse_disjunction_pp3 = 'data/feature_sets/training/incomplete_feature_sets/pp3/train_sbert_infersent_universal_sim_cse_disjunction_pp3.pkl'
train_sbert_infersent_universal_sim_cse_disjunction_tsv_pp3 = 'data/feature_sets/training/incomplete_feature_sets/pp3/train_sbert_infersent_universal_sim_cse_disjunction_pp3.tsv'

## test

test_sbert_infersent_disjunction_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_sbert_infersent_disjunction_pp3.pkl'
test_sbert_infersent_universal_disjunction_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_sbert_infersent_universal_disjunction_pp3.pkl'
test_sbert_infersent_universal_sim_cse_disjunction_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction_pp3.pkl'
test_sbert_infersent_universal_sim_cse_disjunction_tsv_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction_pp3.tsv'

# Combined sentence embeddings similarities + other features

## training

train_first_five_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_five_features_pp3.pkl'
train_first_six_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_six_features_pp3.pkl'
train_first_seven_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_seven_features_pp3.pkl'
train_first_eight_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_eight_features_pp3.pkl'
train_first_nine_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_nine_features_pp3.pkl'
train_first_ten_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_ten_features_pp3.pkl'
train_first_eleven_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_eleven_features_pp3.pkl'
train_first_twelve_features_pp3 = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_twelve_features_pp3.pkl'

train_first_twelve_features_pp3_tsv = 'data/feature_sets/training/pp3/incomplete_feature_sets/train_first_twelve_features_pp3.tsv'

## test

test_first_five_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_five_features_pp3.pkl'
test_first_six_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_six_features_pp3.pkl'
test_first_seven_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_seven_features_pp3.pkl'
test_first_eight_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_eight_features_pp3.pkl'
test_first_nine_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_nine_features_pp3.pkl'
test_first_ten_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_ten_features_pp3.pkl'
test_first_eleven_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_eleven_features_pp3.pkl'
test_first_twelve_features_pp3 = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_twelve_features_pp3.pkl'

test_first_twelve_features_pp3_tsv = 'data/feature_sets/test/pp3/incomplete_feature_sets/test_first_twelve_features_pp3.tsv'

