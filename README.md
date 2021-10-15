# German STT model evaluation

> **tl;dr:** As of October 2021 Jaco-Assistant/Scribosermo model gives you best overall performance.

In search of a "good" STT model in german language I have evaluated all free (as in free beer and open source) models.

Conclusion: both Jaco-Assistant/Scribosermo  models - full and quantized - give you best WER and decent CER at good performance on CPU. 
Wav2Vec is runner up with best CER and good WER, but requires GPU (RTF >1.2 on CPU). Silero is blazing fast but WER of 19% makes it impractical for daily use.


|Vendor / Architecture    |Model    |WER      |CER      |RTF      |Comment  |
|---------|---------|---------:|---------:|---------:|---------|
| [Jaco-Assistant / Scribosermo](https://gitlab.com/Jaco-Assistant/Scribosermo)    |[full](https://www.mediafire.com/folder/jh5unptizgzou/d37cv-wer0066) / Scorer: [D37CV](https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file)    |**9.43**         |3.66         | 0.078        | CPU 8 cores         |
| Jaco-Assistant / Scribosermo    |quantized / Scorer: [D37CV](https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file)    |9.51         |3.70         | 0.096        | CPU 8 cores         |
| [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)   | [deepspeech-german v0.9.0](https://github.com/AASHISHAG/deepspeech-german#trained-models)         |27.93         |11.36         |      0.209   | 
| Mozilla DeepSpeech   | [Polyglot](https://drive.google.com/drive/folders/1oO-N-VH_0P89fcRKWEUlVDm-_z18Kbkb?usp=sharing)         |14.45         |11.36         | 0.241        | 
|[Silero](https://github.com/snakers4/silero-models#silero-models)     |[v4](https://models.silero.ai/models/de/de_v4_large.jit) large     | 18.98        | 6.67        | **0.009**         |  RTF is not a typo       |
| [Wav2Vec](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)    |[jonatasgrosman / wav2vec2-large-xlsr-53-german](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german)     | 10.87        |**2.68**         |   0.06      | Batchsize 1         |
|[Vosk](https://alphacephei.com/vosk/)     | [0.21](https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip)     | 12.84        | 4.56        | 0.292        |         |

For word error rate ([WER](https://huggingface.co/metrics/wer)) and character error rate ([CER](https://huggingface.co/metrics/cer)) the full test-dataset by Huggingface (huggingface/common_voice/de/6.1.0) has been used. 
The real time factor (RTF) has been calculated by running inference on the first 1,000 records of the same dataset as above. Pre-processing times (loading audio files, sample rate conversion, etc.) were excluded.

Evaluation was performed on a [Nvidia Xavier
AGX 32GB](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit) with JetPack 4.6, MAXN mode and jetson-clocks enabled.
