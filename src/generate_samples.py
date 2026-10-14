import os

os.environ["HF_HOME"] = "/home/ewong/scratch/.cache"
"""
Generating sample for Seq-KD using a teacher model.

Prompts come from Hipe-OCRepair 2026 BLOCR Submission by Valentina Vavassori and Harry Lloyd
"""
from pathlib import Path
from typing import Literal
from textwrap import dedent

import pandas as pd
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Sampler:
    def __init__(self, data_dir: str, sample_data_dir: str, model: str,
                 region_col: str, article_col: str) -> None:
        self.data_dir = Path(data_dir)
        self.sample_data_dir = Path(sample_data_dir)
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)
        self.region_col = region_col
        self.article_col = article_col
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_token_length = 0

    def get_max_length(self) -> int:
        """Max article token length across all sampled data."""
        return self.max_token_length

    def _token_lengths(self, texts: pd.Series) -> list[int]:
        enc = self.tokenizer(texts.fillna("").tolist(), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    def create_data_sample(self, fraction_to_sample: float = 0.05) -> None:
        """Sample by region within each yearly file and track max token length."""
        for data_file in sorted(self.data_dir.glob("*.parquet")):
            year = data_file.stem
            df = pd.read_parquet(data_file)
            df = pd.read_parquet(data_file).dropna(subset=[self.article_col])
            sampled_df = df.groupby(self.region_col).sample(frac=fraction_to_sample, random_state=42)

            if not sampled_df.empty:
                lengths = self._token_lengths(sampled_df[self.article_col])
                self.max_token_length = max(self.max_token_length, max(lengths))

            sampled_df = sampled_df[[self.article_col, self.region_col]]
            sampled_df.to_parquet(self.sample_data_dir / f"{year}_sampled.parquet")

class TeacherModel:
    def __init__(self, sample_data_dir: str, output_dir: str, model: str,
                 region_col: str, article_col: str,
                 max_tokens: int, max_model_len: int) -> None:
        self.sample_data_dir = Path(sample_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.region_col = region_col
        self.article_col = article_col
        self.llm = LLM(model=model, max_model_len=max_model_len)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    def _create_message(self, message: str, region: str, time_period: str) -> list[dict[str, str]]:
        """
        Build the chat messages for one OCR post-correction request.

        Notes:
        Modified prompt from https://arxiv.org/html/2607.08143v1 and
        https://github.com/harrylloyd-bl/hipe-ocrepair/blob/main/ocrepair/postcorrect.py
        zero-shot prompt for OCR correction (SYSTEM_PROMPT at commit 5933ceac1a72e8ada03be39bccbe83b107cdf29f).

        Changes from the original:
        - Scope: "newspapers and books ... in English, French and German" -> "newspapers ... in English"
        - Input/output: one article per request returning only the corrected text, instead of a JSON
        array of {document_id, language, ocr_hypothesis} plus a CORRECTIONS_CSV log; rules 1 and 6
        reworded to match ("the text" instead of "ocr_hypothesis")
        - Over-segmentation example: "follow-ing" -> "following". The original output kept a mid-word
        hyphen, which leaves line-break hyphens in the corrected text. Genuine period hyphenation
        (e.g. "to-day") is explicitly preserved.
        - Region/time: publication place and year passed in the user message (system prompt stays
        fixed so it is shared across requests for prefix caching).
        - Rule 3: "within the given language" -> keep English, do not modernize period spelling.
        - Rule 4: accented-character rule replaced with "preserve paragraph breaks".
        - Misrecognized-character examples: "aſter" -> "tbe"/"rnoney" (long s is not period-typical
        for 20th-century US newspapers).
        """
        prompt = dedent("""\
            You are an expert in historical OCR post-correction for English-language newspapers.

            You will receive the publication place, the year, and the raw OCR text of one newspaper article.

            Your task:
            1. Correct OCR errors in the text
            2. Do NOT alter person names or place names
            3. Keep the text in English; do not translate or modernize spelling valid for that place and time
            4. Preserve paragraph breaks
            5. Fix four classes of common historical OCR errors:
            - Over-segmentation: words split across lines, e.g. "before the follow ing morning" -> "before the following morning"
            (keep genuine period hyphenation, e.g. "to-day", "to-morrow")
            - Under-segmentation: missing spaces, e.g. "Rich oldSwarth LAND" -> "Rich old Swarth LAND"
            - Misrecognized character: e.g. "tbe" -> "the"; "ar.d" -> "and"; "rnoney" -> "money"
            - Missing character: e.g. "er a good deal" -> "After a good deal"
            6. If too ambiguous, copy the text unchanged

            Return ONLY the full corrected text, with no explanation, comments or surrounding quotes.
            """)
        user = f"Place: {region}\nYear: {time_period}\n\nOCR text:\n{message}"
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]

    def create_dataset(self) -> int:
        """
        Autocorrect the sampled data with the teacher model.

        Returns the number of samples pruned because their output hit max_tokens.
        """
        pruned = 0
        for sample_file in sorted(self.sample_data_dir.glob("*.parquet")):
            year = sample_file.stem.split("_")[0]
            data_df = pd.read_parquet(sample_file)

            conversations = [
                self._create_message(article, region, year)
                for article, region in zip(data_df[self.article_col], data_df[self.region_col])
            ]

            outputs = self.llm.chat(conversations, sampling_params=self.sampling_params, use_tqdm=True)

            corrected, keep = [], []
            for out in outputs:
                completion = out.outputs[0]
                corrected.append(completion.text.strip())
                keep.append(completion.finish_reason == "stop")

            mask = pd.Series(keep, index=data_df.index)
            data_df["article_spellcheck"] = corrected
            data_df["year"] = year
            pruned += int((~mask).sum())

            child_df = data_df.loc[mask, [self.article_col, "article_spellcheck", self.region_col, "year"]]
            child_df.to_parquet(self.output_dir / f"{year}_corrected.parquet", index=False)

        return pruned

if __name__ == "__main__":
    PROMPT_OVERHEAD = 500
    MODEL = "google/gemma-4-26B-A4B-it"
    SAMPLE_DATA_DIR = "/home/ewong/scratch/distill-ocr/sampled_data"

    data_sample = Sampler(
        data_dir="/home/ewong/scratch/american_stories/merged_data",
        sample_data_dir=SAMPLE_DATA_DIR,
        model=MODEL,
        region_col="State",
        article_col="article",
    )
    data_sample.create_data_sample(0.005)
    longest = data_sample.get_max_length()

    max_tokens = int(longest * 1.3)
    max_model_len = PROMPT_OVERHEAD + longest + max_tokens

    teacher = TeacherModel(
        sample_data_dir=SAMPLE_DATA_DIR,
        output_dir="/home/ewong/scratch/distill-ocr/teacher_training_set",
        model=MODEL,
        region_col="State",
        article_col="article",
        max_tokens=max_tokens,
        max_model_len=max_model_len,
    )
    pruned = teacher.create_dataset()
    print(f"Pruned {pruned} samples that hit max_tokens")