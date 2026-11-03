# IzumxAI-nano: своя модель с нуля (токенизатор -> корпус -> pretrain)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

& $py -m izumxai.fetch_corpus --out data/raw --ru-docs 2000
& $py -m izumxai.train_tokenizer --raw data/raw --out data/tokenizer --vocab 8192
& $py -m izumxai.build_bin --raw data/raw --tokenizer data/tokenizer/tokenizer.json --out data/bin/corpus.bin
& $py -m izumxai.train_pretrain --data data/bin/corpus.bin --tokenizer data/tokenizer/tokenizer.json `
    --preset nano-30m --steps 5000 --batch-size 24 --lr 3e-4 --out runs/nano30m
& $py -m izumxai.chat --ckpt runs/nano30m/latest.pt --tokenizer data/tokenizer/tokenizer.json
