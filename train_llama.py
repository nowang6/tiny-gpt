import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from data import create_dataloader_v1
from lit_gpt.model import GPT, Config
from functools import partial

import wandb


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits.view(-1, logits.size(-1))
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss, batches_seen = 0., 0.
    if num_batches is None:
        num_batches = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_seen += 1
        else:
            break
    return total_loss / batches_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    #context_size = model.pos_emb.weight.shape[0]
    context_size = 64
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()




def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":

    vocab_size = 16002    
    emb_dim = 768
    n_layers = 12
    n_heads = 12
    drop_rate = 0.1
    
    batch_size = 60
    ctx_len = 256
    learning_rate = 5e-4
    weight_decay = 0.1
    num_epochs = 20
    
    wandb.init(
        project="my-llama",

        config={
        "learning_rate": learning_rate,
        "architecture": "transformer",
        "dataset": "CIFAR-100",
        "epochs": num_epochs,
        }
    )
    
    model_name = "tiny_LLaMA_120M"
    config = Config.from_name(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Data
    tokenizer_path = "/home/niwang/data/tokenier/m.model"
    tokenizer = SentencePieceProcessor(model_file = tokenizer_path)
    #data_path = "/home/niwang/data/wiki/wiki_chunk_aa.txt"
    data_path = "/home/niwang/data/xiyou.txt"
    #data_path = "the-verdict.txt"
    with open(data_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        
    
    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        tokenizer,
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=ctx_len,
        stride=ctx_len,
        drop_last=True,
        shuffle=True
    )

    val_loader = create_dataloader_v1(
        tokenizer,
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=ctx_len,
        stride=ctx_len,
        drop_last=False,
        shuffle=False
    )

    ##############################
    # Initialize model
    ##############################
           
    model = GPT(config)
    model.apply(partial(model._init_weights ,n_layer=config.n_layer))
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    
    ###########################
    # Initiate training
    ###########################

    # Main training loop
    tokens_seen, global_step = 0, 0 
    eval_freq = 10
    eval_iter = 1
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in tqdm(train_loader):
            optimizer.zero_grad()  # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        #start_context="Every effort moves you"
        wandb.log({"Train loss ": train_loss, "Val loss": val_loss})
        start_context="孙悟空"
        generate_and_print_sample(
            model, train_loader.dataset.tokenizer, device, start_context
        )
    wandb.finish()
