from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from base_summariser import BaseSummariser


class AbstractiveSummariser(BaseSummariser):

    def __init__(self, text, model_name: str = "google/pegasus-xsum"):
        self.model_name = model_name
        self.text = text
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.torch_device)

    def generate_summary(self):
        tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        batch = tokenizer.prepare_seq2seq_batch(self.text, truncation=True, padding='longest', return_tensors='pt')
        translated = self.model.generate(**batch)
        summary = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return summary


as_obj = AbstractiveSummariser("The big cat was very hungry. It went without food for 24 hours. It lurked all over the town looking for a meal. One day it was a small fish and it brutally killed it. The big cat got some food but it was still hungry. The big cat does not know where to get more fish but it want to eat them. It walks from jungle to jungle looking inside small ponds for more fish.")
print()
