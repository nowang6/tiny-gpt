from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path

data_dir = "/home/niwang/data"
model_path = "/home/niwang/data/tokenier/m"
txt_path = [x for x in Path(data_dir).glob("*.txt")]

SentencePieceTrainer.Train(input=txt_path,
                           vocab_size = 6000,
                           model_prefix = "/home/niwang/data/tokenier/m",
                           model_type = "unigram"
                           )

m = SentencePieceProcessor(model_file = model_path + ".model")

print(m)
