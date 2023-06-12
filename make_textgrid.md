https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-acoustic
copy and edit

pip install -r requirements.txt
git submodule init
git submodule update

python prepare.py

sudo apt install libopenblas-dev libatlas3-base
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
cd montreal-forced-aligner/lib/thirdparty/bin && rm libopenblas.so.0 && ln -s ../../libopenblasp-r0-8dca6697.3.0.dev.so libopenblas.so.0
cd ../../../../

./montreal-forced-aligner/bin/mfa_train_and_align \
data/kss/mfa_input \
data/kss/dict_mfa.txt \
data/kss/mfa_outputs \
-t ./montreal-forced-aligner/kss \
-j 24


(mfa를 돌리고 난 후 ./data/kss/mfa_outputs 에 textgrid가 생성됨)
(이후 textgrid파일을 ./dataset/kss/textgrid 에 넣기)