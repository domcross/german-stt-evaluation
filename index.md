In search of a "good" STT model for German language I have evaluated all free (as in free beer and open source) models.

> **tl;dr** As of December 2021 NeMo-ASRs Conformer-Transducer model is the overall leader (best WER and CER) on GPU, while Jaco-Assistant/Scribosermo model is still a very good choice for CPU.

|Vendor / Architecture    |Model    |WER      |CER      |RTF      |Comment  |
|---------|---------|---------:|---------:|---------:|---------|
| [Jaco-Assistant / Scribosermo](https://gitlab.com/Jaco-Assistant/Scribosermo)    |[full](https://www.mediafire.com/folder/jh5unptizgzou/d37cv-wer0066) / Scorer: [D37CV](https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file)    |_9.43_         |_3.66_         | 0.078        | CPU 8 cores         |
| Jaco-Assistant / Scribosermo    |quantized / Scorer: [D37CV](https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file)    |9.51         |3.70         | 0.096        | CPU 8 cores         |
| [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)   | [deepspeech-german v0.9.0](https://github.com/AASHISHAG/deepspeech-german#trained-models)         |27.93         |11.36         |      0.209   | 
| Mozilla DeepSpeech   | [Polyglot](https://drive.google.com/drive/folders/1oO-N-VH_0P89fcRKWEUlVDm-_z18Kbkb?usp=sharing)         |14.45         |11.36         | 0.241        | 
|[Silero](https://github.com/snakers4/silero-models#silero-models)     |[v4](https://models.silero.ai/models/de/de_v4_large.jit) large     | 18.98        | 6.67        | **0.009**         |  RTF is not a typo       |
| [Wav2Vec](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)    |[jonatasgrosman / wav2vec2-large-xlsr-53-german](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german)     | 10.87        |2.68         |   0.06      | Batchsize 1         |
|[Vosk](https://alphacephei.com/vosk/)     | [0.21](https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip)     | 12.84        | 4.56        | 0.292        |         |
|[Nvidia NeMo-ASR](https://github.com/NVIDIA/NeMo)     | [Conformer-CTC 1.5.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_conformer_ctc_large/)     | 7.39        | 1.80        | 0.064        | GPU w/[Apex-AMP](https://github.com/NVIDIA/apex)       |
|Nvidia NeMo-ASR     | [Conformer-Transducer 1.5.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_conformer_transducer_large)     | **6.20**        | **1.62**        | 0.124        | GPU w/Apex-AMP      |
|Nvidia NeMo-ASR     | [Citrinet-1024 1.5.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_citrinet_1024)     | 8.24        | 2.32        | 0.069        | GPU w/Apex-AMP      |
|Nvidia NeMo-ASR     | [Contextnet-1024 1.4.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_contextnet_1024)     | 6.68        | 1.77        | 0.098        | GPU w/Apex-AMP      |
|Nvidia NeMo-ASR     | [Quartznet-15x15 1.0.0rc1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_quartznet15x5)     |13.23        | 3.53        | 0.064        | GPU w/Apex-AMP      |


### Conclusion
For GPU NeMo-ASRs models are leader of the pack. The Conformer-Transducer model gives you best WER and CER, the Contextnet-1024 and Conformer-CTC models are runner up with still very good values and even better RTF than the Transducer model. 

On CPU both Jaco-Assistant/Scribosermo  models - full and quantized - give you good WER/CER values and good performance. (Note: Jaco website claims WER 7.5% while I got "only" 9.4%).
Silero is blazing fast but WER of 19% makes it impractical for daily use.

#### Notes on methodology
Word error rate ([WER](https://huggingface.co/metrics/wer)) and character error rate ([CER](https://huggingface.co/metrics/cer)) were calculated (with PyPi-package jiwer==2.2.0) on the Common-Voice test-dataset provided by Huggingface (huggingface/common_voice/de/6.1.0 retrieved with PyPi-package datasets==1.13.3). 
The real time factor (RTF) has been calculated by running inference on the first 1,000 records of the same dataset as above. Pre- and post-processing times (loading audio files, sample rate conversion, normalizing results, etc.) were excluded.

Evaluation was performed on a [Nvidia Xavier
AGX 32GB](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit) with JetPack 4.6, MAXN mode and jetson-clocks enabled.

You like this page? Then don't be shy and go to the repository and click the star-button: <a class="github-button" href="https://github.com/domcross/german-stt-evaluation" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star domcross/german-stt-evaluation on GitHub">Star</a>

<script async defer src="https://buttons.github.io/buttons.js"></script>
