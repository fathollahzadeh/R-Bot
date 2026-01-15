#!/bin/bash

cd rag
unzip stackoverflow-rewrite-embed.zip
# Build structure-semantics Q&A index.
python3 rag_gen.py
# (Optional) Build structure-only Q&A index.
python3 rag_structure.py
# (Optional) Build semantics-only Q&A index.
python3 rag_semantics.py


 rm -rf venv
 python3.10 -m venv venv
 source venv/bin/activate
 pip install --upgrade pip
 pip install -r requirements.txt


