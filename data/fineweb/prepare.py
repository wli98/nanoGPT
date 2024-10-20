# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import concurrent.futures

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 30
bs = 32
#num_val_rows = 5000//32
num_val_rows = 5000
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    def tokenize_single(text):
        toks = enc.encode_ordinary(text)
        toks.append(enc.eot_token)
        return toks 

    def tokenize_mp(example):
        example_strings = example['text']
        with concurrent.futures.ProcessPoolExecutor() as executor:
            tokenized_strings = list(executor.map(tokenize_single, example_strings))
        example['toks'] = tokenized_strings
        example['len'] = [len(toks) for toks in tokenized_strings]
        return example
    def tokenize_multiple(example):
        example_strings = example['text']
        tokenized_strings = []
        for string in example_strings:
            toks = enc.encode_ordinary(string)
            toks.append(enc.eot_token)
            tokenized_strings.append(toks)
        example['toks'] = tokenized_strings
        example['len'] = [len(toks) for toks in tokenized_strings]
        return example
    
    def tokenize_nobatch(example):
        example_string = example['text']
        toks = enc.encode_ordinary(example_string)
        toks.append(enc.eot_token)
        example['toks'] = toks 
        example['len'] = len(toks) 
        return example

    #dataset = dataset.map(tokenize_mp,batched=True)
    dataset = dataset.map(tokenize_nobatch)
    #dataset = dataset.take(15000)
    #dataset = dataset.batch(batch_size=500)

    # owt by default only contains the 'train' split, so create a test split
    val_len = 0
    train_len = 0
    #iterate first time to get length
    for i,entry in tqdm(enumerate(dataset)):
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        if i < num_val_rows:
           # val_len += sum(entry['len'])
            val_len += entry['len']
        else:
           # train_len += sum(entry['len'])
            train_len += entry['len']
 
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    train_filename = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_filename = os.path.join(os.path.dirname(__file__), 'val.bin')
    train_arr = np.memmap(train_filename, dtype=dtype, mode='w+', shape=(train_len,))
    val_arr = np.memmap(val_filename, dtype=dtype, mode='w+', shape=(val_len,))
    
    val_idx = 0
    train_idx = 0
    for i,entry in tqdm(enumerate(dataset)):
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        tokens = entry['toks']
        if i < num_val_rows:
            val_arr[val_idx:val_idx+len(tokens)] = tokens 
            val_idx += len(tokens)
        else:
            train_arr[train_idx:train_idx+len(tokens)] = tokens 
            train_idx += len(tokens)
    train_arr.flush()
    val_arr.flush()
    ''' 
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        avg_len = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            avglen = batch['len'].mean()
            if batch_idx == 0:
                avg_len = avglen
            else:
                avg_len =((avg_len)*batch_idx + avglen)/(batch_idx+1)
            print(avg_len)

        arr.flush()
    '''
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
