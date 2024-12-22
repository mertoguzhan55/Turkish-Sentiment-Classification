from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataclasses import dataclass
from app.logger import Logger


@dataclass
class Inference:

    logger: Logger
    input_text: str
    model_dir: str
    max_length: int
    truncation: bool

    def preprocess_text(self,text, tokenizer):
        """ Preprocess text for inference. """
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return inputs

    def predict(self, text, model, tokenizer, label_map):
        """ Perform inference on a single text input. """
        torch.cuda.empty_cache()

        inputs = self.preprocess_text(text, tokenizer)

        # Move model and inputs to the same device
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device("cpu")
        model = model.to(device)  
        inputs = {key: value.to(device) for key, value in inputs.items()} 

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

        predicted_label = [key for key, value in label_map.items() if value == predicted_class_idx][0]

        return predicted_label
    
    def infer(self):
        label_map = {"kızgın": 0, "korku": 1, "mutlu": 2, "surpriz": 3, "üzgün": 4}

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir,num_labels=5)

        predicted_label = self.predict(self.input_text, model, tokenizer, label_map)
        print(predicted_label)



if __name__ == "__main__":
    pass
