import pandas as pd
import json
import ast
import spacy
import os

nlp = spacy.load("en_core_web_sm")

def extract_conversations(output_file):
    previous_conversation_index = None
    conversation_list = []

    with open(output_file, 'r') as file:
        next(file)  # skip the header row
        rows = file.readlines()

        for row in rows:
            if row[0].isdigit():  # skip the sentence rows
                current_conversation_index = int(row.split("\t")[0])

                if current_conversation_index != previous_conversation_index:
                    # start a new conversation if the conversation_index is different
                    conversation_list.append(row.strip())
                    previous_conversation_index = current_conversation_index
                else:
                    # append the sentence to the current conversation if the conversation_index is the same
                    conversation_list[-1] += " " + row.strip()
    
    return conversation_list



def tsv_add_id(tsv_file):
    '''
    This function reads a tsv file and adds conversation, sentence, and token IDs to each word in the file.
    Conversation ID is added to the first word of each conversation. Sentence ID is added to the first word of each sentence within the same conversation. Token ID is added to each word within the same sentence.
    '''

    
    data_with_id = []
    conversation = 1
    

    with open(tsv_file, 'r') as file:
        next(file)  # Skip the "utterance" line
        sentence_id = 1  # Initialize sentence_id
        for line in file:
            line = line.strip()

            if line.endswith('--------------------'):
                conversation += 1
                data_with_id.append(('', '', '', ''))  # Add an empty line to separate conversations
                sentence_id = 1  # Reset sentence_id for new conversation
            elif line:  # Check if the line is not empty
                # Add a sentence to make the file more readable for annotations
                sentence = line.split('\t')[0]
                # data_with_id.append((sentence, '', '', ''))

                doc = nlp(sentence)  # Use spaCy to process the sentence
                token_id = 1
                for token in doc:
                    data_with_id.append((str(conversation), str(sentence_id), str(token_id), token.text))  # Add conversation, sentence, and token IDs to each word
                    token_id += 1  # Increment token_id
                sentence_id += 1  # Increment sentence_id
                data_with_id.append(('', '', '', ''))  # Add an empty line to separate sentences

    return data_with_id


def get_label(df, activity_dict):
    # Iterate through the dataframe:
    activity_dict = {k: ast.literal_eval(v) for k, v in activity_dict.items()}
    for index, row in df.iterrows():
        
        conversation_key = f'conversation_{row["conversation"]}'  # Make the key to search in the activity_dict

        if conversation_key in activity_dict:
            activities = activity_dict[conversation_key]

            for activity in activities:
                # see if there is a key named 'time_sentence_id' in the activity
                if 'time_sentence_id' not in activity:
                    continue
                    
                # Initialize label as 'O'
                label = 'O'
                
                if int(activity['activity_sentence_id']) == int(row['sent_id']) and int(row['token_id']) in activity['activity_token_ids']:
                    if int(row['token_id']) == activity['activity_token_ids'][0]:
                        label = 'B-event'
                    else:
                        label = 'I-event'
                

                elif activity['time_sentence_id'] != 'None' and activity['time_sentence_id'] is not None and int(activity['time_sentence_id']) == int(row['sent_id']) and int(row['token_id']) in activity['time_token_ids']:
                    if int(row['token_id']) == activity['time_token_ids'][0]:
                        label = 'B-time'   
                    else:
                        label = 'I-time'


                elif activity['place_sentence_id'] != 'None' and activity['place_sentence_id'] is not None and int(activity['place_sentence_id']) == int(row['sent_id']) and int(row['token_id']) in activity['place_token_ids']:
                    if int(row['token_id']) == activity['place_token_ids'][0]:
                        label = 'B-place'
                    else:
                        label = 'I-place'


                elif activity['participants_sentence_id'] != 'None' and activity['participants_sentence_id'] is not None and int(activity['participants_sentence_id']) == int(row['sent_id']) and int(row['token_id']) in activity['participants_token_ids']:
                    if int(row['token_id']) == activity['participants_token_ids'][0]:
                        label = 'B-participants'
                    else:
                        label = 'I-participants'
                
                # Determine the correct event column based on activity_index
                event_column = f'event{activity["activity_index"]}'

                # Assign the label to the correct event column
                if label != 'O':
                    df.at[index, event_column] = label
                    break
    
    return df


# Convert to jsonl format BY CONVERSATION
def tsv_to_jsonl(filepath, category):
    # Read the TSV file
    df = pd.read_csv(filepath, sep='\t')

    # Group by conversation
    grouped = df.groupby('conversation')

    # List to store JSON objects
    json_list = []

    # Iterate through each conversation group
    for conversation_id, group in grouped:
        # List of words in the conversation
        seq_words = group['token'].tolist()

        # Iterate through each event column
        for event_col in ['event1', 'event2', 'event3', 'event4', 'event5']:
            # List of BIO labels in the event column
            BIO = group[event_col].tolist()
            
            #  Check if there are any non-'O' labels in the BIO list
            if any(label != 'O' for label in BIO):
                # Create a JSON object
                json_obj = {
                    "seq_words": seq_words,
                    "BIO": BIO,

                    # Not nessary for the current task
                    # "pred_sense": [str(category)], 

                    "src_lang": "<EN>"
                }
                
                # Append the JSON object to the list
                json_list.append(json_obj)

                
    output_jsonl = filepath.replace('.tsv', '.jsonl')

    # Write the JSON objects to a JSONL file
    with open(output_jsonl, 'w') as jsonl_file:
        for json_obj in json_list:
            jsonl_file.write(json.dumps(json_obj) + '\n')



# Convert to jsonl format BY SENTENCE
def convert_tsv_to_jsonl(tsv_dir, jsonl_dir):
    # Iterate over all files in the root directory
    for file_name in os.listdir(tsv_dir):
        # Check if the file is a TSV file
        if file_name.endswith('.tsv') and "annotation" not in file_name:
            # Read the TSV file
            input_file = os.path.join(tsv_dir, file_name)
            df = pd.read_csv(input_file, sep='\t')

            # Identify all columns that start with "event"
            event_columns = [col for col in df.columns if col.startswith('event')]

            # Function to create the 'label' column
            def get_label(row):
                labels = []
                for col in event_columns:
                    if pd.notna(row[col]) and row[col] != 'O':
                        labels.append(row[col])
                return labels[0] if labels else 'O'

            # Apply the get_label function to generate the 'label' column
            df['label'] = df.apply(get_label, axis=1)

            # Drop the original event columns
            df.drop(columns=event_columns, inplace=True)

            # Save the new TSV file
            output_file = os.path.join(jsonl_dir, file_name.replace('.tsv', '.jsonl'))
            df.to_csv(output_file, sep='\t', index=False)

            # Read the modified TSV file
            df = pd.read_csv(output_file, sep='\t')

            # Group rows by conversation and sentence
            grouped = df.groupby(["conversation", "sent_id"])

            # category = file_name.split('/')[-1].split('_')[0]

            # Function to construct JSONL format
            def construct_jsonl(row):
                seq_words = list(row["token"])
                BIO = list(row["label"])

                # Not nessary for the current task
                # pred_sense = [str(category)]
                
                src_lang = "<EN>"
                return {"seq_words": seq_words, "BIO": BIO, "src_lang": src_lang}

            # Iterate over grouped DataFrame and construct JSONL for each group
            jsonl_data = []
            for name, group in grouped:
                jsonl_data.append(construct_jsonl(group))

            # Write the JSONL data to a file
            with open(output_file, "w") as f:
                for item in jsonl_data:
                    json.dump(item, f)
                    f.write("\n")


def convert_single_tsv_to_jsonl(tsv_dir, output_file):

    df = pd.read_csv(tsv_dir, sep='\t')

    # Identify all columns that start with "event"
    event_columns = [col for col in df.columns if col.startswith('event')]

    # # Function to create the 'label' column
    # def get_label(row):
    #     labels = []
    #     for col in event_columns:
    #         if pd.notna(row[col]) and row[col] != 'O':
    #             labels.append(row[col])
    #     return labels[0] if labels else 'O'

    # # Apply the get_label function to generate the 'label' column
    # df['label'] = df.apply(get_label, axis=1)

    # # Drop the original event columns
    # df.drop(columns=event_columns, inplace=True)

    # Save the new TSV file
    df.to_csv(output_file, sep='\t', index=False)

    # Read the modified TSV file
    df = pd.read_csv(output_file, sep='\t')

    # Group rows by conversation and sentence
    grouped = df.groupby(["conversation", "sent_id"])

    # category = file_name.split('/')[-1].split('_')[0]

    # Function to construct JSONL format
    def construct_jsonl(row):
        seq_words = list(row["token"])
        BIO = list(row["label"])

        # Not nessary for the current task
        # pred_sense = [str(category)]
        
        src_lang = "<EN>"
        return {"seq_words": seq_words, "BIO": BIO, "src_lang": src_lang}

    # Iterate over grouped DataFrame and construct JSONL for each group
    jsonl_data = []
    for name, group in grouped:
        jsonl_data.append(construct_jsonl(group))

    # Write the JSONL data to a file
    with open(output_file, "w") as f:
        for item in jsonl_data:
            json.dump(item, f)
            f.write("\n")
