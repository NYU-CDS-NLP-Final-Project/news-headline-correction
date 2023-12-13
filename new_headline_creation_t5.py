import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model1(article, max_len=256):
    model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
    tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
    model = model.to(device)

    text =  "headline: " + article

    max_len = max_len

    encoding = tokenizer.encode_plus(text, return_tensors = "pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids = input_ids,
        attention_mask = attention_masks,
        max_length = 64,
        num_beams = 3,
        early_stopping = True,
    )
    result = tokenizer.decode(beam_outputs[0])
    
    return result
