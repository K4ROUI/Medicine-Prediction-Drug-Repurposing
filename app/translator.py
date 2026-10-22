# translator.py (optional file)
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, src_lang='en', tgt_lang='fr'):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=texts, return_tensors="pt", padding=True)
        translated = self.model.generate(**batch)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
