# AI Research Assistant (RAG)

A Python and Streamlit demo that indexes local text and Markdown documents, retrieves relevant passages with TF-IDF, and displays source labels for manual review.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests -v
```

Add documents you are permitted to use under `data/documents/`. Similarity scores are ranking signals, not truth scores. Check important claims against original sources. The local demo does not implement authentication or production access controls.