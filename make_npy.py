import tgt
import os
import numpy as np
from tqdm import tqdm

dataset_path = './dataset/kss'
textgrid_path = f'{dataset_path}/textgrid'

result_path = f'{dataset_path}/result'
phone_path = f'{result_path}/phone'
duration_path = f'{result_path}/duration'

sample_rate = 22050
hop_length = 256

if not os.path.exists(result_path):
    os.mkdir(result_path)

if not os.path.exists(phone_path):
    os.mkdir(phone_path)

if not os.path.exists(duration_path):
    os.mkdir(duration_path)


def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for i, t in enumerate(tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                phones.append('s')
                duration_cal = int(e*sample_rate/hop_length)-int(s*sample_rate/hop_length)
                durations.append(duration_cal)
                continue
            else:
                phones.append('s')
                durations.append(0)
                start_time = s

        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            continue
        
        duration_cal = int(e*sample_rate/hop_length)-int(s*sample_rate/hop_length)
        
        if i+1 != len(tier._objects):
            n_t = tier._objects[i+1]
            n_s, n_e, n_p = n_t.start_time, n_t.end_time, n_t.text
            if n_p in sil_phones:
                duration_cal += int(n_e*sample_rate/hop_length)-int(n_s*sample_rate/hop_length)
        durations.append(duration_cal)

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, np.array(durations)

for textgrid_name in tqdm(os.listdir(textgrid_path)):
    base_name = textgrid_name.replace('.TextGrid', '')
    textgrid = tgt.io.read_textgrid(f'{textgrid_path}/{textgrid_name}')
    phone, duration = get_alignment(textgrid.get_tier_by_name('phones'))
    
    with open(f'{phone_path}/{base_name}.txt', 'w') as f:
        f.write("".join(phone).replace('SEP', ' ').replace('PUNC', '.'))

    np.save(f'{duration_path}/{base_name}.wav.npy', duration, allow_pickle=False)