from torch.utils.data import Dataset, DataLoader
import torch
from sentencepiece import SentencePieceProcessor

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(tokenizer, txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader


if __name__ == '__main__':
    tokenizer_path = "/home/niwang/data/tokenier/m.model"
    tokenizer = SentencePieceProcessor(model_file = tokenizer_path)
    
    file_path = "/home/niwang/data/wiki/wiki_chunk_aa.txt"
    with open(file_path, "r",encoding='utf-8') as file:
        text_data = file.read()
        
    dataloader = create_dataloader_v1(tokenizer=tokenizer,txt=text_data)
    print(dataloader)