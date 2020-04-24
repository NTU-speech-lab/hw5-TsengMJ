mkdir -p Model

wget -P ./Model https://github.com/NTU-speech-lab/hw3-TsengMJ/releases/download/0/best_sgdm_700.pth

python ./Src/hw5.py "$1" "$2" ./Model/best_sgdm_700.pth