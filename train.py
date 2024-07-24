import torch
import torch.nn as nn

from dataset import BillingualDataset
from model import build_transformer
from config import get_config,get_weights_file_path
 
from torch.utils.data import DataLoader,Dataset,random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
 
def get_all_sentences(ds,lang):
     for item in ds:
         yield item["translation"][lang]
 
def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[EOS]","[SOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw=load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split='train')
    
    #Build tokenizer
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])
    
    #Keep 90% for training 10% for validation
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])
    
    train_ds=BillingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds=BillingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    
    max_len_src=0
    max_len_tgt=0
    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids=tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))
    
    train_dataloader=DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=config["batch_size"],shuffle=True)
    
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(src_vocab_size=vocab_src_len,tgt_vocab_size=vocab_tgt_len,src_seq_len=config["seq_len"],tgt_seq_len=config["seq_len"])
    return model

def train_model(config):
    device="cuda"
    Path(config["model_folder"]).mkdir(parents=True,exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    
    #Tensorboard
    writer=SummaryWriter(config["experiment_name"])
    
    optimiser=torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    
    initial_epoch=0
    global_step=0
    
    if config["preload"]:
        model_filename=get_weights_file_path(config,config["preload"])
        print(f"Preloading model: {model_filename}")
        state=torch.load(model_filename)
        initial_epoch=state["epoch"]+1
        optimiser.load_state_dict(state["optimizer_state_dict"])
        global_step=state["global_step"]
        
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch,config["num_epochs"]):
        model.train()
        batch_iterator=tqdm(train_dataloader,desc=f"Processing epoch {epoch}")
        for batch in batch_iterator:
            
            encoder_input=batch['encoder_input'].to(device) #(batch,seq_len)
            decoder_input=batch['decoder_input'].to(device) #(batch,seq_len)
            encoder_mask=batch['encoder_mask'].to(device)  #(batch,1,1,seq_len)
            decoder_mask=batch['decoder_mask'].to(device)  #(batch,1,seq_len,seq_len)
            
            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output=model.project(decoder_output) #(batch,seq_len,tgt_vocab_size)
            
            label=batch["label"].to(device)#(batch,seq_len)
            
            #(batch,seq_len,tgt_vocab_size)->(batch*seq_len,tgt_vocab_size)
            loss =loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch.iterator.set_postfix(f"loss:{loss.item():6.3f}")
            
            #log the loss
            writer.add_scalar("train loss",loss.item(),global_step)
            writer.flush()
            
            #Backpropagate loss
            loss.backward()
            
            #update the weights
            optimiser.step()
            optimiser.zero_grad()
            
            global_step+=1
        
        #Save the model
        model_filename=get_weights_file_path(config,f"{epoch:02d}")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimiser_state_dict':optimiser.state_dict(),
            'global_step':global_step
        },model_filename)
        
        
        
if __name__=="__main__":
    config=get_config()
    train_model(config)
            
            