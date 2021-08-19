#%%
# https://www.youtube.com/watch?v=cHymMt1SQn8
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
import os
import re
from tika import parser
from nltk import sent_tokenize
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_metric
import numpy as np
#%%
# Get the Training data and assemble it into one text document
report_dir = 'IPCC_reports'
file_list = os.listdir(report_dir)
print(file_list)
all_reports_text = []
for file in file_list:
    sentences = sent_tokenize(parser.from_file('IPCC_reports/'+file)['content'])
    i = 0
    for s in list(sentences):
        sentences[i] = re.sub('\\n',"", s)
        all_reports_text.append(sentences[i])
        i=i+1
    
print(all_reports_text[:5])

#%%
# Create the Training and eval datasets from sentences:
train, eval = train_test_split(all_reports_text, train_size=.9, random_state=2020)
print("training size:",len(train))
print("Evaluation size: ", len(eval))
with open('train_tmp.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(train))

with open('eval_tmp.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(eval))

#%%
# run the fine tune script:
# load the tokenizer (what translates text to numbers) andthe model (brains, I guess)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
training_args = TrainingArguments("train_tmp.txt")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train_tmp.txt"],
    eval_dataset=tokenized_datasets["eval_tmp.txt"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
predictions = trainer.predict(tokenized_datasets["eval_tmp.txt"])
trainer.train()

#%%
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# %%
sentence = 'The climate is changing'
# converts the sentence into numerical inputs
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Decode and generate text

output = model.generate(input_ids, max_length = 100, num_beams=5, no_repeat_ngram_size = 3, early_stopping = True)

tokenizer.decode(output[0], skip_special_tokens=True)



# %%
# Output result

text = tokenizer.decode(output[0], skip_special_tokens=True)

