# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:02:06 2021

@author: Shadow
"""

from datasets import load_dataset, load_metric 
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import default_data_collator
from tqdm.auto import tqdm
import numpy as np
import collections
from collections import Counter


def tokenize_squad(this_example, this_tokenizer, subdataset_name=None, **kwargs
):
        
    # The maximum length of a feature (question and context)
    max_length = kwargs['max_seq_length'] if 'max_seq_length' in kwargs else 384
    
    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = kwargs['doc_stride'] if 'doc_stride' in kwargs else 128 
    
    #For this function to work with any kind of models, 
    #we need to account for the special case where the model expects padding on the left 
    #(in which case we switch the order of the question and the context)
    pad_on_right = this_tokenizer.padding_side == "right"
    
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    this_example["question"] = [q.lstrip() for q in this_example["question"]]
    
    
    
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = this_tokenizer(
        this_example["question" if pad_on_right else "context"],
        this_example["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples["offset_mapping"]
    
    #we need the example_id for testing
    tokenized_examples["example_id"] = []

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(this_tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        
        #so example['answers'] will return a list of answers for each example
        #we index examples["answers"] by sample_index so we can find the answer associated with this example
        answers = this_example["answers"][sample_index]
        tokenized_examples['example_id'].append(this_example['id'][sample_index])

        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def new_postprocess_qa_predictions(tokenized_test_data, raw_predictions, this_tokenizer, **kwargs):
    
    assert "squad_v2" in kwargs, "You must specify if you are using squad_v2"
    
    squad_v2 = kwargs['squad_v2']
    n_best_size = kwargs['n_best_size'] if 'n_best_size' in kwargs else 20
    
    # The authorized overlap between two part of the context when splitting it is needed.
    max_answer_length = kwargs['max_answer_length'] if 'max_answer_length' in kwargs else 30 
    
    tokenized_test_data.set_format(type=tokenized_test_data.format["type"], columns=list(tokenized_test_data.features.keys()))    
    
    
    
    all_start_logits, all_end_logits = raw_predictions
        
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(tokenized_test_data["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(tokenized_test_data):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        


    # The dictionaries we have to fill.
    #predictions = collections.OrderedDict()
    predictions = []

    # Let's loop over all the examples!
    for example_index in range(len(tqdm(tokenized_test_data))):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        #context = example["context"]
        input_ids = tokenized_test_data[example_index]['input_ids']
        #print('input ids len: ', len(input_ids))
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = tokenized_test_data[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = tokenized_test_data[feature_index]["input_ids"].index(this_tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    #start_char = offset_mapping[start_index][0]
                    #end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": this_tokenizer.decode(input_ids[start_index: end_index+1])
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            #predictions[example_index["id"]] = best_answer["text"]
            predictions.append(best_answer["text"])
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            #predictions[example_index["id"]] = answer
            predictions.append(answer)

    return predictions


def main(squad_v2=False):
    
    #im trying to use less data here to see if this works
    #datasets = load_dataset("squad_v2" if squad_v2 else "squad")
    
    train_data = load_dataset("squad_v2" if squad_v2 else "squad", split = 'train[:5%]')
    val_data = load_dataset("squad_v2" if squad_v2 else "squad", split = 'validation[:10%]')
    test_data = load_dataset("squad_v2" if squad_v2 else "squad", split = 'validation[10%:20%]')
    

    #model_checkpoint = 'facebook/muppet-roberta-base'
    model_checkpoint = 'distilbert-base-uncased'
    batch_size = 16
    
        
    #global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    tokenized_train = train_data.map(lambda example: tokenize_squad(example, this_tokenizer = tokenizer), batched=True, remove_columns=train_data.column_names)
    tokenized_val = val_data.map(lambda example: tokenize_squad(example, this_tokenizer = tokenizer), batched=True, remove_columns=val_data.column_names)
    tokenized_test = test_data.map(lambda example: tokenize_squad(example, this_tokenizer = tokenizer), batched=True, remove_columns=test_data.column_names)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-squad",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01)
    
    
    data_collator = default_data_collator
    
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    raw_predictions = trainer.predict(tokenized_test)
    
    postprocess_kwargs = {'squad_v2':False}
    
    final_predictions = new_postprocess_qa_predictions(tokenized_test, raw_predictions.predictions, this_tokenizer=tokenizer, **postprocess_kwargs)
    
    print('final preds: ', final_predictions)
    
if __name__ == "__main__":
    main()