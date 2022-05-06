import enum
import pandas as pd

from src.pre_processing.src.cleaning_text.sequence_cleaner import SequenceCleaner
from src.pre_processing.src.finding_claim_context.context_finder import ContextFinder


class PreprocessingMethod(enum.Enum):
    cleaning_tweets = 1
    simple_context = 2
    said_context = 3
    speaker_context = 4


class PreProcessor:

    def __init__(self, preprocessing_method):
        if preprocessing_method == PreprocessingMethod.cleaning_tweets.name:
            self.preprocessing_method = PreprocessingMethod.cleaning_tweets.name
        elif preprocessing_method == PreprocessingMethod.simple_context.name:
            self.preprocessing_method = PreprocessingMethod.simple_context.name
        elif preprocessing_method == PreprocessingMethod.said_context.name:
            self.preprocessing_method = PreprocessingMethod.said_context.name
        elif preprocessing_method == PreprocessingMethod.speaker_context_context.name:
            self.preprocessing_method = PreprocessingMethod.speaker_context_context.named
        else:
            raise ValueError('Choose preprocessing method.')

    def pre_process(self, input_file, output_file):
        if self.preprocessing_method == PreprocessingMethod.cleaning_tweets.name:
            preprocessor = SequenceCleaner('full_cleaning')
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                cleaned_i_claim = preprocessor.clean(i_claim)
                with open(output_file, 'a', encoding='utf-8') as f:
                    joined_list = "\t".join([i_claim_id, cleaned_i_claim])
                    print(joined_list, file=f)
        else:
            context_finder = ContextFinder(self.preprocessing_method)
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                contextualized_i_claim = context_finder.contextualize(i_claim_id)
                with open(output_file, 'a', encoding='utf-8') as f:
                    joined_list = "\t".join([i_claim_id, contextualized_i_claim])
                    print(joined_list, file=f)

