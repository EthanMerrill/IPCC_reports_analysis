#%%
# https://www.youtube.com/watch?v=cHymMt1SQn8
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
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

