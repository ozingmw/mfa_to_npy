import tgt
import os
import librosa
import soundfile as sf
import tqdm

dataset_path = './dataset/kss'
textgrid_path = f'{dataset_path}/textgrid'
wavs_path = f'{dataset_path}/wavs'
wavs_cut_path = f'{dataset_path}/wavs_cut'

if not os.path.exists(wavs_cut_path):
    os.mkdir(wavs_cut_path)

for textgrid_name in tqdm.tqdm(os.listdir(textgrid_path)):
    textgrid = tgt.io.read_textgrid(f'{textgrid_path}/{textgrid_name}')
    tier = textgrid.get_tier_by_name('phones')
    end_time = float(tier._objects[-1].end_time)
    
    wav_name = textgrid_name.replace('.TextGrid', '.wav')

    y, sr = librosa.load(f'{wavs_path}/{wav_name}', sr=22050)
    ny = y[:int(sr*end_time)]
    sf.write(f'{wavs_cut_path}/{wav_name}', ny, sr)