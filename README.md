# distill-ocr

A prototype pipeline that distills an LLM OCR spellchecker (the teacher) into a
smaller student model, for cheaper spell-checking of historical newspapers at
similar accuracy.

> **Status: prototype. Not recommended for use.**
>
> The student does **not** beat the HIPE-OCRepair random benchmark, even with a
> strong teacher. A larger student than Qwen 0.6B is probably needed.

## Overview

1. **Sample** noisy OCR text from historical newspapers.
2. **Correct** it with a teacher LLM, prompted with a prompt adapted from the
   British Library's HIPE-OCRepair submission, to produce (noisy → corrected)
   pairs.
3. **Fine-tune** a small student model (Qwen 0.6B) on those pairs.
4. **Evaluate** the student against HIPE-OCRepair.

## Data & Attribution

This project uses the datasets below. They remain the property of their
creators and are used under the terms of their licenses. No endorsement by the
dataset authors is implied.

### American Stories (U.S. newspapers)

- **Source:** [dell-research-harvard/AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
  (DOI: [10.57967/hf/0757](https://doi.org/10.57967/hf/0757))
- **Authors:** Dell et al. (2023)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This
  license permits copying, redistribution, adaptation and commercial use as
  long as the creators are credited, the license is linked, and changes are
  stated.
- **Underlying scans:** Library of Congress,
  [Chronicling America](https://chroniclingamerica.loc.gov/).
- **Changes made:** articles were sampled and corrected by a teacher LLM to
  produce training pairs for the student model.

### British Library HIPE-OCRepair submission (teacher prompt)

- **Source:** [harrylloyd-bl/hipe-ocrepair](https://github.com/harrylloyd-bl/hipe-ocrepair),
  the British Library team's contribution to HIPE-OCRepair 2026
- **Authors:** Valentina Vavassori and Harry Lloyd (British Library)
- **License:** [MIT](https://github.com/harrylloyd-bl/hipe-ocrepair/blob/main/LICENSE)
- **Changes made:** the teacher prompt in this repo is adapted from the BL
  team's submission.

MIT notice for the adapted material:

```
Copyright (c) <YEAR> <COPYRIGHT HOLDER>   <- copy this line from their LICENSE file

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### HIPE-OCRepair 2026 (evaluation)

- **Benchmark:** [HIPE-OCRepair 2026](https://hipe-eval.github.io/HIPE-OCRepair-2026/)
  ([data](https://github.com/hipe-eval/HIPE-OCRepair-2026-data),
  [scorer](https://github.com/hipe-eval/HIPE-OCRepair-scorer))
- See the HIPE-OCRepair data repository for the licenses of its component
  datasets.

## Citation

If you use this work, please cite the datasets it builds on:

```bibtex
@article{dell2023american,
  title   = {American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers},
  author  = {Dell, Melissa and Carlson, Jacob and Bryan, Tom and Silcock, Emily and Arora, Abhishek and Shen, Zejiang and D'Amico-Wong, Luca and Le, Quan and Querubin, Pablo and Heldring, Leander},
  journal = {arXiv preprint arXiv:2308.12477},
  year    = {2023}
}

@misc{vavassori2026blhipe,
  title        = {{BL} contribution to the 2026 {HIPE-OCRepair} competition},
  author       = {Vavassori, Valentina and Lloyd, Harry},
  year         = {2026},
  howpublished = {\url{https://github.com/harrylloyd-bl/hipe-ocrepair}}
}

@article{ehrmann2026hipeocrepair,
  title   = {{ICDAR} 2026 {HIPE-OCRepair} Competition on {LLM}-Assisted {OCR} Post-Correction for Historical Documents},
  author  = {Ehrmann, Maud and Boros, Emanuela and Opitz, Juri and Michail, Andrianos and Wagner, Florian and Clematide, Simon},
  journal = {arXiv preprint arXiv:2607.08143},
  year    = {2026}
}
```
