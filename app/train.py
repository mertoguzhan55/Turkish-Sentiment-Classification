from dataclasses import dataclass
from datasets import load_dataset
from app.logger import Logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,  # CausalLM yerine SequenceClassification kullanıldı
                          TrainingArguments, 
                          Trainer)


@dataclass
class TurkishClassification:

    logger: Logger
    model_name: str
    num_labels: int
    load_in_4bit: bool
    rank: int  
    lora_alpha: int
    lora_dropout: float
    bias: str
    task_type: str  
    train_dir: str
    val_dir: str
    model_output_dir: str
    evaluation_strategy: str
    learning_rate: float
    per_device_train_batch_size: int  
    per_device_eval_batch_size: int
    num_train_epochs: int
    weight_decay: float
    save_strategy: str
    logging_dir: str
    logging_steps: int
    save_total_limit: int
    fp16: bool 

    def __post_init__(self):
        pass
    
    def preprocess_labels(self, examples):
        label_map = {"kızgın": 0, "korku": 1, "mutlu": 2, "surpriz": 3, "üzgün": 4}
        
        texts = []
        processed_labels = []
        
        for idx, (text, label) in enumerate(zip(examples['Tweet'], examples['label'])):

            if text is None or not isinstance(text, str) or text.strip() == "":
                continue
                
            label = label.strip().lower()
            if label not in label_map:
                print(f"Hata Satırı {idx + 1}: Geçersiz etiket: {label}")
                continue
                
            texts.append(text)
            processed_labels.append(label_map[label])
        
        return {
            'Tweet': texts,
            'label': processed_labels
        }
    
    def preprocess_data(self, examples, tokenizer):
        return tokenizer(
            examples['Tweet'], 
            truncation=True, 
            padding='max_length', 
            max_length=128
        )

    def train(self):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,  
            quantization_config={"load_in_4bit": self.load_in_4bit},  # 4-bit quantization
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.rank,  # Lora rank
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],  # layers which will be optimize
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type, 
        )

        model = get_peft_model(model, lora_config)

        dataset = load_dataset('csv', data_files={'train': self.train_dir, 'test': self.val_dir})

        dataset = dataset.map(
            self.preprocess_labels,
            remove_columns=dataset['train'].column_names, 
            batched=True
        )

        # tokenize dataset
        encoded_dataset = dataset.map(
            self.preprocess_data, 
            batched=True, 
            fn_kwargs={'tokenizer': tokenizer}
        )

        # 5. Fine-Tuning settings
        training_args = TrainingArguments(
            output_dir=self.model_output_dir,
            evaluation_strategy=self.evaluation_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size, 
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            save_strategy=self.save_strategy,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,  # for usage less memory
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
        )

        # training
        trainer.train()

        # save the model
        trainer.save_model(self.model_output_dir)
        tokenizer.save_pretrained(self.model_output_dir)
