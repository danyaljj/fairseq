
##1) get the data 
```shell script
> ls 
boolq-dev.jsonl
boolq-train.jsonl
```

##2)


##3) BPE Encode
Run `multiprocessing_bpe_encoder`, you can also do this in previous step for each sample but that might be slower.

```shell script
# Download encoder.json and vocab.bpe
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

for SPLIT in train dev; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "examples/roberta/boolq-dev.jsonl" \
        --outputs "examples/roberta/boolq-$SPLIT.bpe" \
        --workers 60 \
        --keep-empty
done
```


##4) Preprocess data
# Download fairseq dictionary.

```shell script
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.input0.bpe" \
    --validpref "aclImdb/dev.input0.bpe" \
    --destdir "IMDB-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.label" \
    --validpref "aclImdb/dev.label" \
    --destdir "IMDB-bin/label" \
    --workers 60
```

