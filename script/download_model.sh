# Huggingface Downloader

echo "Downloading [$1]..."
echo "Cache Path: [~/hf_cache/]"

huggingface-cli download $1 --cache-dir ~/hf_cache/