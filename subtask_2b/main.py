import pandas as pd

from src.evaluation.scorer.main import evaluate_CLEF
from src.feature_generation import complete_feature_set_pairs_train, complete_feature_set_pairs_test, test_data_pp1
from src.feature_generation.feature_set_generator import FeatureSetGenerator
from src.feature_generation.file_paths.TEST_file_names import complete_feature_set_pairs_test_TEST
from src.feature_generation.unsupervised_feature_set_generator import UnsupervisedFeatureSetGenerator
from src.pre_processing.pre_processor import PreProcessor
from src.prediction.predictor import Predictor

top_5_sbert = 'data/unsupervised_ranking/pp1/top_5_sbert.tsv'
top_5_universal = 'data/unsupervised_ranking/pp1/top_5_universal.tsv'
top_5_infersent = 'data/unsupervised_ranking/pp1/top_5_infersent.tsv'
top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sim_cse.tsv'
top_5_seq_match = 'data/unsupervised_ranking/pp1/top_5_seq_match.tsv'
top_5_levenshtein = 'data/unsupervised_ranking/pp1/top_5_levenshtein.tsv'
top_5_jacc_chars = 'data/unsupervised_ranking/pp1/top_5_jacc_chars.tsv'
top_5_jacc_tokens = 'data/unsupervised_ranking/pp1/top_5_jacc_tokens.tsv'
top_5_ne = 'data/unsupervised_ranking/pp1/top_5_ne.tsv'
top_5_main_syms = 'data/unsupervised_ranking/pp1/top_5_main_syms.tsv'
top_5_words = 'data/unsupervised_ranking/pp1/top_5_words.tsv'
top_5_subjects = 'data/unsupervised_ranking/pp1/top_5_subjects.tsv'
top_5_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_ne_ne_ratio.tsv'
top_5_ne_token_ratio = 'data/unsupervised_ranking/pp1/top_5_ne_token_ratio.tsv'
top_5_main_syms_ratio = 'data/unsupervised_ranking/pp1/top_5_main_syms__ratio.tsv'
top_5_main_syms_token_ratio = 'data/unsupervised_ranking/pp1/top_5_main_syms_token_ratio.tsv'
top_5_words_ratio = 'data/unsupervised_ranking/pp1/top_5_words_ratio.tsv'
top_5_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_words_token_ratio.tsv'
top_5_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok.tsv'
top_5_sim_cse_jacc_tok_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_words.tsv'
top_5_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words.tsv'
top_5_sim_cse_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_ne.tsv'
top_5_sim_cse_jacc_tok_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_ne.tsv'
top_5_all_features = 'data/unsupervised_ranking/pp1/top_5_all_features.tsv'
top_5_all_features_without_infersent = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
top_5_no_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_no_sentence_embeddings.tsv'
top_5_sbert_universal_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse.tsv'
top_5_sbert_universal_sim_cse_ne_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_features.tsv'
top_5_sbert_universal_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_jacc_tok.tsv'
top_5_all_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_all_sentence_embeddings.tsv'
top_5_sbert_universal_sim_cse_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_ne_ratio_features.tsv'
top_5_sbert_universal_sim_cse_ne_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_main_syms_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_main_syms_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words.tsv'
top_5_sbert_infersent_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words.tsv'
top_5_sbert_infersent_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse.tsv'
top_5_sbert_infersent_sim_cse_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words_token_ratio.tsv'
top_5_sbert_sim_cse_features = 'data/unsupervised_ranking/pp1/top_5_sbert_sim_cse.tsv'
top_5_sbert_infersent_sim_cse_words_token_ratio_words = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words_token_ratio_words.tsv'
top_5_sbert_infersent = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent.tsv'
top_5_infersent_universal_sim_cse = 'data/unsupervised_ranking/pp1/top_5_infersent_universal_sim_cse.tsv'
top_5_universal_sim_cse = 'data/unsupervised_ranking/pp1/top_5_universal_sim_cse.tsv'

if __name__ == '__main__':

    # ## pp1
    #
    # training_data = 'data/original_speech_data/training_data/CT2022-Task2B-EN-Train-Dev_Queries.tsv'
    # test_data = 'data/original_speech_data/test_data/queries.tsv'
    # training_data_labels_train = 'data/original_speech_data/training_data/CT2022-Task2B-EN-Train_QRELs.tsv'
    # training_data_labels_dev = 'data/original_speech_data/training_data/CT2022-Task2B-EN-Dev_QRELs.tsv'
    # all_training_data_labels = 'data/original_speech_data/training_data/all_train.pkl'
    # test_data_labels = 'data/original_speech_data/test_data/task2b-test.tsv'
    # v_claims = 'data/politifact-vclaims'
    #
    #
    # pp1_classification = 'data/predictions/pp1/classification.tsv'
    # pp1_classification_incomplete = 'data/predictions/pp1/classification_incomplete.tsv'
    #
    #
    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio'])
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(complete_feature_set_pairs_train, complete_feature_set_pairs_test, test_data_pp1, pp1_classification)
    # evaluate_CLEF(test_data_labels, pp1_classification) # 0.4335 # try again with different heuristic sim score

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id','sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(complete_feature_set_pairs_test)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id','sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(training_df, test_df, test_data_pp1, pp1_classification)
    # evaluate_CLEF(test_data_labels, pp1_classification) # 0.4262


    # # ufsg = UnsupervisedFeatureSetGenerator(['words_token_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_words_token_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_words_token_ratio) #  0.2399
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['words_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_words_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_words_ratio) # 0.2599
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_token_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_main_syms_token_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_main_syms_token_ratio) #0.1814
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_main_syms_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_main_syms_ratio) #0.2127
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['ne_token_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_ne_token_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_ne_token_ratio) #0.1424
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['ne_ne_ratio'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_ne_ne_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_ne_ne_ratio) #0.1344
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['subjects'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_subjects, n=5)
    # evaluate_CLEF(test_data_labels, top_5_subjects) #0.0570
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['words'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_words, n=5)
    # evaluate_CLEF(test_data_labels, top_5_words) #0.2753
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['main_syms'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_main_syms, n=5)
    # evaluate_CLEF(test_data_labels, top_5_main_syms) #0.1993
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['ne'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_ne, n=5)
    # evaluate_CLEF(test_data_labels, top_5_ne) #0.1380
    #
    # # ufsg = UnsupervisedFeatureSetGenerator(['jacc_tokens'], 'pp1')
    # # ufsg.create_top_n_output_file(test_data, top_5_jacc_tokens, n=5)
    # evaluate_CLEF(test_data_labels, top_5_jacc_tokens) #0.2321
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['jacc_chars'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_jacc_chars, n=5)
    # evaluate_CLEF(test_data_labels, top_5_jacc_chars) #0.0468
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['levenshtein'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_levenshtein, n=5)
    # evaluate_CLEF(test_data_labels, top_5_levenshtein) #0.1477
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['seq_match'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_seq_match, n=5)
    # evaluate_CLEF(test_data_labels, top_5_seq_match) #0.2295

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_all_sentence_embeddings, n=5)
    # evaluate_CLEF(test_data_labels, top_5_all_sentence_embeddings) #0.4061

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sbert, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sbert) # 0.2979
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['infersent'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_infersent, n=5)
    # evaluate_CLEF(test_data_labels, top_5_infersent) # 0.2253
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['universal'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_universal, n=5)
    # evaluate_CLEF(test_data_labels, top_5_universal) # 0.3481
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sim_cse, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sim_cse) #0.3516

    # ufsg = UnsupervisedFeatureSetGenerator(['infersent', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_infersent_universal_sim_cse, n=5)
    # evaluate_CLEF(test_data_labels, top_5_infersent_universal_sim_cse) #0.4008

    # ufsg = UnsupervisedFeatureSetGenerator(['universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_universal_sim_cse, n=5)
    # evaluate_CLEF(test_data_labels, top_5_universal_sim_cse) #0.4135






    # fsg.generate_feature_set(test_data)
    # fsg.generate_feature_set(training_data, all_training_data_labels)

    # labels = fsg.combine_labels(training_data_labels_train, training_data_labels_dev, all_training_data_labels)
    # labels = all_training_data_labels
    # featureset_train = fsg.generate_feature_set(training_data, labels)

    # featureset_train = complete_feature_set_triples_train+'.pkl'
    # featureset_test = complete_feature_set_triples_test+'.pkl'
    #
    # predictor = Predictor('triple_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_triple)
    # evaluate_CLEF(test_data_labels, predictions_triple) #MAP@5 0.3734, PRECISION@1 0.3797 = MRR

    # featureset_train = complete_feature_set_pairs_train
    # featureset_test = complete_feature_set_pairs_test

    # predictor = Predictor('binary_proba')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_binary_proba)
    # evaluate_CLEF(test_data_labels, predictions_binary_proba) #MAP@5 0.3608, PRECISON@1 0.3671 = MRR

    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_binary)
    # evaluate_CLEF(test_data_labels, predictions_binary) #MAP@5 0.6915, PRECISION@1 0.3797, MRR 0.3998 #lga auch nicht an n=1

    # featureset_train = complete_feature_set_pairs_train
    # featureset_test = complete_feature_set_pairs_test
    #
    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_highest_50_se_sims, n=50)
    # evaluate_CLEF(test_data_labels, predictions_highest_50_se_sims) #MAP@5 0.4232, PRECISION@1 0.3797, MRR 0.4307

    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_highest_10_se_sims, n=10)
    # evaluate_CLEF(test_data_labels, predictions_highest_10_se_sims) #MAP@5 0.4232, PRECISION@1 0.3797, MRR 0.4227

    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_highest_5_se_sims)
    # evaluate_CLEF(test_data_labels, predictions_highest_5_se_sims) #MAP@5 0.4232, PRECISION@1 0.3797, MRR 0.4196

    # predictor = Predictor('triple_classification_with_rank_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, predictions_triple_double_classification)
    # evaluate_CLEF(test_data_labels, predictions_triple_double_classification) #MAP@5 0.4190, PRECISION@1 0.3797, MRR 0.4158

    # fsg = FeatureSetGenerator(['main_syms_ratio', 'words_ratio'])
    # # fsg.prepare_vclaims(v_claims)
    # fsg.generate_feature_set(test_data)

    top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sim_cse.tsv'
    top_5_sbert = 'data/unsupervised_ranking/pp1/top_5_sbert.tsv'
    top_5_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok.tsv'
    top_5_sim_cse_jacc_tok_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_words.tsv'
    top_5_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words.tsv'
    top_5_sim_cse_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_ne.tsv'
    top_5_sim_cse_jacc_tok_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_ne.tsv'
    top_5_all_features = 'data/unsupervised_ranking/pp1/top_5_all_features.tsv'
    top_5_all_features_without_infersent = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    top_5_no_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_no_sentence_embeddings.tsv'
    top_5_sbert_universal_sim_cse_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_features.tsv'
    top_5_sbert_universal_sim_cse_ne_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_features.tsv'
    top_5_sbert_universal_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_jacc_tok.tsv'
    top_5_all_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_all_sentence_embeddings.tsv'
    top_5_sbert_universal_sim_cse_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_ne_ratio_features.tsv'
    top_5_sbert_universal_sim_cse_words_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_ratio_features.tsv'
    top_5_sim_cse_words_ratio = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words_ratio.tsv'
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sbert_universal_sim_cse_features, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sbert_universal_sim_cse_features) #0.4129

    # ufsg = UnsupervisedFeatureSetGenerator(['sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sim_cse, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sim_cse) # 0.3516
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sbert, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sbert) #0.2979
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'words_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sbert_universal_sim_cse_words_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sbert_universal_sim_cse_words_ratio) # 0.3951
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sim_cse', 'words_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(test_data, top_5_sim_cse_words_ratio, n=5)
    # evaluate_CLEF(test_data_labels, top_5_sim_cse_words_ratio) # 0.3854



    # FeatureSelector.feature_correlation(complete_feature_set_pairs_train, feature_correlation_training_data_spearman)
    # FeatureSelector.mutual_information_feature_selection(complete_feature_set_pairs_train)

    # top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sim_cse.tsv'
    # top_5_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok.tsv'
    # top_5_sim_cse_jacc_tok_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_words.tsv'
    # top_5_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words.tsv'
    # top_5_sim_cse_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_ne.tsv'
    # top_5_sim_cse_jacc_tok_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_ne.tsv'
    # top_5_all_features = 'data/unsupervised_ranking/pp1/top_5_all_features.tsv'
    # top_5_all_features_without_infersent = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    # top_5_sim_cse_ = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    # top_5_no_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_no_sentence_embeddings.tsv'
    #
    # ranker = UnsupervisedRanker(['sim_cse'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse) #0.3516
    #
    # ranker = UnsupervisedRanker(['sim_cse','jacc_tok'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse_jacc_tok)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse_jacc_tok) #0.3834
    #
    # ranker = UnsupervisedRanker(['sim_cse','jacc_tok', 'ne'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse_jacc_tok_ne)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse_jacc_tok_ne) #0.3907
    #
    # ranker = UnsupervisedRanker(['sim_cse', 'ne'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse_ne)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse_ne) #0.3653
    #
    # ranker = UnsupervisedRanker(['sim_cse','jacc_tok', 'words'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse_jacc_tok_words)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse_jacc_tok_words) #0.3967
    #
    # ranker = UnsupervisedRanker(['sim_cse', 'words'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_sim_cse_words)
    # evaluate_CLEF(test_data_labels,  top_5_sim_cse_words) #0.3797
    #
    # ranker = UnsupervisedRanker(['sbert', 'infersent', 'universal', 'sim_cse', 'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_all_features)
    # evaluate_CLEF(test_data_labels,  top_5_all_features) #0.1850
    #
    # ranker = UnsupervisedRanker(['sbert', 'universal', 'sim_cse', 'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_all_features_without_infersent)
    # evaluate_CLEF(test_data_labels,  top_5_all_features_without_infersent) #0.1882

    # ranker = UnsupervisedRanker(
    #     ['sim_cse', 'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns',
    #      'words', 'subjects'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, test_data, top_5_no_sentence_embeddings)
    # evaluate_CLEF(test_data_labels,  top_5_no_sentence_embeddings) #0.1850



    # ## TEST ##
    #
    # output = 'data/output/subtask2B_english.tsv'
    #
    # test_data = 'data/original_speech_data/TEST/test_TEST.tsv'
    #
    # predictions_binary_TEST = 'data/predictions/TEST/binary.tsv'
    #
    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                             'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                             'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                                                       'words_token_ratio'])
    #
    # # featureset_test = fsg.generate_feature_set(test_data)
    # featureset_test = complete_feature_set_pairs_test_TEST
    #
    # featureset_train = complete_feature_set_pairs_train
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, test_data, output)




    ## pp2 said context normal vclaims

    training_data = 'data/original_speech_data/training_data/CT2022-Task2B-EN-Train-Dev_Queries.tsv'
    test_data = 'data/original_speech_data/test_data/queries.tsv'
    pp2_test_data = 'data/pp_speech_data/test_data/pp2/claims_test_said_context_1_pp2.tsv'
    pp2_training_data = 'data/pp_speech_data/training_data/pp2_iclaims.queries'

    # pre_processor = PreProcessor('said_context')
    # pp2_training_data = training_data
    # pp2_test_data = pre_processor.pre_process(test_data, pp2_test_data)

    fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
                               'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
                               'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
                               'words_token_ratio'])
    fsg = FeatureSetGenerator(['main_syms_ratio', 'main_syms_token_ratio', 'w