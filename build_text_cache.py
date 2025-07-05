import json, numpy as np, torch, os, argparse
from sentence_transformers import SentenceTransformer  # 约 110 MB，轻量
                                            # 或改用 fastText/BioWordVec

parser = argparse.ArgumentParser()
parser.add_argument('--prompt_json', default='prompts.json')
parser.add_argument('--out', default='text_feats.npy')
args = parser.parse_args()

model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')  # 医学文本
prompts = json.load(open(args.prompt_json, 'r', encoding='utf8'))
keys, feats = [], []
for k, sent in prompts.items():
    keys.append(k)
    feats.append(model.encode(sent, normalize_embeddings=True))  # L2 归一化
np.save(args.out, np.vstack(feats))
json.dump(keys, open(args.out.replace('.npy', '_keys.json'), 'w'))
print('>>> saved', args.out, 'shape=', np.vstack(feats).shape)
