## Llama-index demo with OCP docs and groq

### Prerequistes

Python requirements

```shell
pip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface gradio
```

Export Groq API key

```shell
export export GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### Index PDFs

```shell
python index.py
```

### Test it

```shell
python starter.py "Does ROSA support STS ?"
```

### Chat with gradio

```shell
python chat.py
```
