# Bilingual Essay NLP Analyser

## Run locally
Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

HanLP downloads its Chinese tokenizer and CTB9 POS model on the first Chinese run. Upload official NGSL/NAWL CSV or Excel lists in the sidebar for vocabulary coverage.

## Design notes
- There is no app-level cap on the number of files, but infrastructure limits still apply.
- One POS worksheet is created per input file; duplicate/invalid Excel sheet names are handled.
- The metrics workbook has a blank row, mean row, and sample-SD row after the file rows.
- English syntactic metrics use spaCy dependencies. Chinese NP/clause-family metrics are transparent rule-based estimates, not gold-standard parses. Validate them before research publication.
- The supplied notebook was incomplete and contained repeated Colab fragments, so this is a clean reimplementation rather than a line-by-line port.
