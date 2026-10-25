from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset


class StudentModel:
    def __init__(self, data_dir: str, model: str, weight_store_dir: str):
        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        # Full fine-tune: every parameter is trainable, no adapters
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto")
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.weight_store_dir = Path(weight_store_dir)
        self.weight_store_dir.mkdir(exist_ok=True, parents=True)

    def tokenize(self, example):
        """
        Builds input + EOS + target and masks the input so loss is only on the corrected text.
        EOS separates input from output; at inference feed article + EOS and generate until EOS.
        """
        eos = self.tokenizer.eos_token_id
        prompt_ids = self.tokenizer(example["article"], add_special_tokens=False)["input_ids"] + [eos]
        target_ids = self.tokenizer(example["article_spellcheck"], add_special_tokens=False)["input_ids"] + [eos]

        return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}

    def finetune(self):
        """
        Supervised fine-tuning of the student on the teacher's corrections
        """
        dataset = load_dataset(
            "csv",
            data_files={
                "train": [str(f) for f in self.train_dir.glob("*.csv")],
                "val": [str(f) for f in self.val_dir.glob("*.csv")],
            },
        )
        dataset = dataset.map(self.tokenize, remove_columns=["article", "article_spellcheck"])

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
        )

        training_args = TrainingArguments(
            output_dir=str(self.weight_store_dir),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(str(self.weight_store_dir / "final"))
        self.tokenizer.save_pretrained(str(self.weight_store_dir / "final"))


if __name__ == "__main__":
    student = StudentModel(
        data_dir="/home/ewong/scratch/distill-ocr/teacher_dataset",
        model="Qwen/Qwen3-0.6B-Base", 
        weight_store_dir="/home/ewong/scratch/distill-ocr/student_weights",
    )
    student.finetune()
