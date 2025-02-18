import yaml
from transformers import AutoTokenizer, AutoModel
from utils import hiragana2IPA, phonemize
from datasets import load_dataset
import os
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import gc
import ast
from simple_loader import FilePathDataset, build_dataloader
from tqdm import tqdm
import pickle

config_path = "Configs/config_ja.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))

tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer']) # you can use any other tokenizers if you want to

dataset = load_dataset("wiki40b", "ja", split="train") # you can use other version of this dataset


def decode_text(sample_text):
    """
    Chuyển đổi chuỗi có dạng b'...' thành chuỗi UTF-8.
    
    Args:
        sample_text (str): Chuỗi có thể chứa biểu diễn bytes dạng string.
        
    Returns:
        str: Chuỗi đã giải mã UTF-8, hoặc giữ nguyên nếu không cần giải mã.
    """
    if isinstance(sample_text, str):
        if (sample_text.startswith("b'") and sample_text.endswith("'")) or (sample_text.startswith('b"') and sample_text.endswith('"')):
            try:
                sample_text = ast.literal_eval(sample_text)  # Chuyển từ string thành bytes thực sự
                sample_text = sample_text.decode("utf-8")   # Giải mã bytes thành UTF-8
            except (SyntaxError, ValueError):
                pass  # Trả về chuỗi gốc nếu gặp lỗi
    return sample_text

def clean_wiki_text(text):
    text = ''.join(text.split('_START_PARAGRAPH_')[1:])
    markers = ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_", "_NEWLINE_", "_START_HEADING_", 
               "_START_BULLET_", "_START_LIST_", "_START_TABLE_", "_START_CAPTION_", "_START_IMAGE_"]
    for marker in markers:
        text = text.replace(marker, "")
    return text.strip()

def preprocess_text(text):
    text = decode_text(text)
    text = clean_wiki_text(text)
    return text

root_directory = "./wiki_phoneme"
num_shards = 1000
processed_shards = []
def process_shard(i):
    directory = root_directory+"/shard_" + str(i)
    if os.path.exists(directory):
        # print("Shard %d already exists!" % i)
        return
    print('Processing shard %d ...' % i)
    try:
        shard = dataset.shard(num_shards=num_shards, index=i)
        processed_dataset = shard.map(lambda t: phonemize(preprocess_text(t['text']), tokenizer), remove_columns=['text'])
        if not os.path.exists(directory):
            os.makedirs(directory)
        processed_dataset.save_to_disk(directory)
        processed_shards.append(i)
        print(f'Shard {i} processed successfully.')
        del processed_dataset  # Free memory
        gc.collect()
    except Exception as e:
        print(f'Error processing shard {i}: {e}')

num_cores = os.cpu_count()
max_workers = num_cores #min(num_cores, 2)

with ProcessPool(max_workers=max_workers) as pool:
    pool.map(process_shard, range(num_shards), timeout=120)

# for i in range(num_shards):
#     if i not in processed_shards:
#         # print(f'Processing shard {i} again...')
#         process_shard(i)
# a = preprocess_text(dataset.shard(num_shards=num_shards, index=19)[517]['text'])
# print(a)
# b = phonemize(a, tokenizer)
# print(b)

output = [dI for dI in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory,dI))]

from datasets import load_from_disk, concatenate_datasets

datasets = []

for o in output:
    directory = f"{root_directory}/{o}"
    try:
        shard = load_from_disk(directory)
        datasets.append(shard)
        print("%s loaded" % o)
    except:
        continue

dataset = concatenate_datasets(datasets)
dataset.save_to_disk(config['data_folder'])
print('Dataset saved to %s' % config['data_folder'])

file_data = FilePathDataset(dataset)
loader = build_dataloader(file_data, num_workers=32, batch_size=128)

special_token = config['dataset_params']['word_separator']

unique_index = [special_token]
for _, batch in enumerate(tqdm(loader)):
    unique_index.extend(batch)
    unique_index = list(set(unique_index))

token_maps = {}
for t in tqdm(unique_index):
    word = tokenizer.decode([t])
    token_maps[t] = {'word': word, 'token': unique_index.index(t)}

with open(config['dataset_params']['token_maps'], 'wb') as handle:
    pickle.dump(token_maps, handle)
print('Token mapper saved to %s' % config['dataset_params']['token_maps'])