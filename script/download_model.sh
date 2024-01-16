# Huggingface Downloader

echo "Downloading [$1]..."
echo "Cache Path: [/raid/hpc/mingzhe/transformers_cache]"

huggingface-cli download $1 --cache-dir /raid/hpc/mingzhe/transformers_cache