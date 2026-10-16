# End-to-End Text-to-Speech Synthesis (UE24CS352A Mini-Project)

Akula Pragnya (PES1UG24AM028) · Akshata Amara (PES1UG24AM025)
UE24CS352A Machine Learning — Mini-Project · October 2026

Replication of *End-to-End Text to Speech Synthesis* (CS 229, Stanford, Autumn 2018):
text -> spectrogram with three models (SVR, simple neural net, seq2seq + attention),
then spectrogram -> wav. See `TTS_Mini_Project_Report.pdf` for the full write-up and
`TTS_Project_Overview.pptx` for the presentation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download LJ Speech (https://keithito.com/LJ-Speech-Dataset/), extract it so you have
`LJSpeech-1.1/metadata.csv` and `LJSpeech-1.1/wavs/`, and either place that folder in the
project root or set `LJ_DIR` to its location.

## Run

```bash
python preprocess.py                       # 200 short clips -> data/dataset.npz
python postprocess.py --roundtrip          # sanity check: real audio -> spec -> wav

python train.py --model svr                # add --kernel linear for the linear variant
python train.py --model nn                 # embedding + sigmoid hidden layer
python train.py --model seq2seq            # LSTM encoder / attention decoder

python synthesize.py "hello world" --model all    # wavs -> outputs/
python eval/mos.py --init                  # then fill eval/mos.csv from listener ratings
python eval/mos.py                         # mean opinion score per model
```

Training loss curves and held-out MSE are written to `results/`.

## Pipeline

1. **Text**: words -> CMU dictionary phonemes or letters (random per word), symbol ids, zero padded.
2. **Audio**: FIR pre-emphasis filter -> STFT (n_fft 2048, 1025 bins) -> log magnitude scaled to [0,1], zero padded.
3. **Models**: SVR (poly/linear), MLP, seq2seq with additive attention.
4. **Postprocess**: inverse log scaling -> Griffin-Lim -> inverse pre-emphasis -> wav.

## Deviations from the paper

- Griffin-Lim is used for phase reconstruction (only magnitudes are predicted).
- SVR predicts 32 PCA components per timestep instead of 1025 outputs per timestep (training time).
- Clips are limited to 3.5 s / 50 characters so all models train on a laptop.

## Results

10 held-out sentences, log-magnitude spectrograms scaled to [0,1]:

| Model                                | Final train loss | Validation MSE | Training time |
|---------------------------------------|-------------------|------------------|----------------|
| Mean-spectrogram baseline             | 0.00361           | 0.00305          | –              |
| SVR (polynomial kernel, PCA 32)       | 0.00218           | 0.00323          | 22 s           |
| Neural network (200 epochs)           | 0.00050           | 0.00327          | 608 s          |
| Seq2seq with attention (150 epochs)   | 0.00105           | 0.00344          | 4,640 s        |

All three models are slightly worse than the trivial mean-spectrogram baseline on
held-out validation MSE; see `TTS_Mini_Project_Report.pdf` for analysis.
