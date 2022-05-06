import enum
import os
import pandas as pd

from src.pre_processing.src.finding_claim_context import TRANSCRIPT_DIR


class PreprocessingMethod(enum.Enum):
    cleaning_tweets = 1
    simple_context = 2
    said_context = 3
    speaker_context = 4


class ContextFinder:

    def __init__(self, context_method, n=1):
        if context_method == PreprocessingMethod.simple_context.name:
            self.context_method = PreprocessingMethod.simple_context.name
            self.n = n
        elif context_method == PreprocessingMethod.said_context.name:
            self.context_method = PreprocessingMethod.said_context.name
            self.n = n
        elif context_method == PreprocessingMethod.speaker_context.name:
            self.context_method = PreprocessingMethod.speaker_context.name

    @staticmethod
    def get_transcript_file_and_line_number_from_id(i_claim_id):
        underscore_positions = []
        for i in range(len(i_claim_id)):
            if i_claim_id[i] == '_':
                underscore_positions.append(i)
        if underscore_positions:
            last_hyphen = underscore_positions[len(underscore_positions) - 1]
            line_number = i_claim_id[last_hyphen+1:]
            file_name = i_claim_id[:last_hyphen]+'.tsv'
            file_and_line_tuple = (line_number, file_name)
        return file_and_line_tuple

    def contextualize(self, i_claim_id):
        print(i_claim_id)
        transcript_file_and_line_tuple = self.get_transcript_file_and_line_number_from_id(i_claim_id)
        file_path = transcript_file_and_line_tuple[1]
        line_number = transcript_file_and_line_tuple[0]
        print(file_path)
        if os.path.isfile(TRANSCRIPT_DIR+'/'+file_path):
            transcript_df = pd.read_csv(TRANSCRIPT_DIR+'/'+file_path, sep='\t', lineterminator='\n', quoting=3, names=['line_number', 'name', 'claim'], encoding='utf-8', dtype=str)
            test = False
        else:
            transcript_df = pd.read_csv(TRANSCRIPT_DIR+'/'+file_path, sep='\t', lineterminator='\n', quoting=3, names=['line_number', 'name', 'claim'], encoding='utf-8', dtype=str)
            test = True
        if self.context_method == PreprocessingMethod.simple_context.name:
            list_of_line_numbers = []
            line_int = int(line_number)
            for i in range(line_int - self.n, line_int + self.n + 1):
                if test:
                    list_of_line_numbers.append(str(i))
                else:
                    list_of_line_numbers.append(str(i).zfill(4))
            output_claim = ''
            sub_transcript = transcript_df.loc[transcript_df['line_number'].isin(list_of_line_numbers)]
            for row in sub_transcript.iterrows():
                output_claim = output_claim + row[1][2] + ' '
            return output_claim
        if self.context_method == PreprocessingMethod.said_context.name:
            list_of_line_numbers = []
            line_int = int(line_number)
            for i in range(line_int - self.n, line_int + self.n + 1):
                if test:
                    list_of_line_numbers.append(str(i))
                else:
                    list_of_line_numbers.append(str(i).zfill(4))
            output_claim = ''
            sub_transcript = transcript_df.loc[transcript_df['line_number'].isin(list_of_line_numbers)]
            for row in sub_transcript.iterrows():
                name = row[1][1].lower()
                claim = row[1][2]
                output_claim = output_claim + name + ' said "' + claim + '" '
            return output_claim
        if self.context_method == PreprocessingMethod.speaker_context.name:
            one_speaker = True
            name = transcript_df.loc[transcript_df['line_number'] == line_number]['name'].values[0]
            list_of_line_numbers = [line_number]
            while(one_speaker):
                if test:
                    lower_line_number = str(int(line_number)-1)
                else:
                    lower_line_number = str(int(line_number)-1).zfill(4)
                try:
                    if transcript_df.loc[transcript_df['line_number'] == lower_line_number]['name'].values[0] == name:
                        list_of_line_numbers.append(lower_line_number)
                        line_number = lower_line_number
                    else:
                        one_speaker = False
                except:
                    one_speaker = False
            one_speaker = True
            line_number = transcript_file_and_line_tuple[0]
            while(one_speaker):
                if test:
                    higher_line_number = str(int(line_number) + 1)
                else:
                    higher_line_number = str(int(line_number)+1).zfill(4)
                try:
                    if transcript_df.loc[transcript_df['line_number'] == higher_line_number]['name'].values[0] == name:
                        list_of_line_numbers.append(higher_line_number)
                        line_number = higher_line_number
                    else:
                        one_speaker = False
                except:
                    one_speaker = False
            list_of_line_numbers.sort(key=int)
            output_claim = ''
            sub_transcript = transcript_df.loc[transcript_df['line_number'].isin(list_of_line_numbers)]
            for row in sub_transcript.iterrows():
                output_claim = output_claim + row[1][2].replace('"',"\"") + ' '
            return output_claim
