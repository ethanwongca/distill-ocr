from pathlib import Path
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer 

class CreateStudentModel:
    def __init__(self, data_dir: str, model: str, weight_store_dir: str):
        self.data_dir = Path(data_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto")
        self.weight_store_dir = Path(weights_store_dir)
        self.weight_store_dir.mkdir(exist_ok=True, parents=True)

    def set_training_args(**huggingface_training_args):
        """
        Wrap around HF training args to fine-tune
        """
        training_args = TrainingArguments(
            output_dir=self.weight_store_dir,
            **huggingface_training_args
        )

        return training_args