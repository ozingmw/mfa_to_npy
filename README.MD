PREPROCESSING AND GENERATE wav to duration

1. wav, textgrid 파일 업로드
kss 음성 데이터를 wavs안에 집어 넣기<br/>
./data/kss/wavs/1_0001.wav<br/>
metadata.csv 파일 넣기 (이때 txt파일을 확장자만 csv로 변경하여 넣는다.)<br/>
./data/kss/metadata.csv<br/>
1_0001|안녕하세요.|안녕하세요.<br/>
textgrid에 필요한 문장은 0,2번째

1. make textgrid
make_textgrid에 있는 prompt를 순서대로 실행

2. make npy
python make_npy.py

3. cut wavs
python cut_wavs.py

4. wavs_cut 폴더를 naturalspeech 폴더로 옮기기
