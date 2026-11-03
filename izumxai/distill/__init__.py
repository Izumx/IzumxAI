"""Дистилляция: данные от учителя -> QLoRA -> GGUF/ollama.

Модули импортируются явно, чтобы `python -m izumxai.distill.gen_data`
не тянул torch, transformers и bitsandbytes.
"""
