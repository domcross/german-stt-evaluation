'''
download modeles and scorer files and put them in subfolder "model":

Quartznet15x5, CV only (WER: 7.5%): https://www.mediafire.com/folder/rrse5ydtgdpvs/cv-wer0077
Quartznet15x5, D37CV (WER: 6.6%):  https://www.mediafire.com/folder/jh5unptizgzou/d37cv-wer0066
Scorer: 
TCV    https://www.mediafire.com/file/xb2dq2roh8ckawf/kenlm_de_tcv.scorer/file
D37CV  https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file
PocoLg https://www.mediafire.com/file/b64k0uqv69ehe9p/de_pocolm_large.scorer/file

'''
import warnings
warnings.filterwarnings('ignore')

import re
import random
import json
import multiprocessing as mp
import time

from datasets import load_dataset, load_metric
from tqdm import tqdm
import numpy as np
import tflite_runtime.interpreter as tflite

# If you want to improve the transcriptions with an additional language model, without using the
# training container, you can find a prebuilt pip-package in the published assets here:
# https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
# or for use on a Raspberry Pi you can use the one from extras/misc directory
from ds_ctcdecoder import Alphabet, Scorer, ctc_beam_search_decoder

import librosa

LANG_ID = "de"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ",
                   "'","-"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test") #[:125]
wer = load_metric("wer")
cer = load_metric("cer")

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


### prepare jaco/scriobosermo model

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    batch["sentence"] = batch["sentence"].replace("Ä","AE")
    batch["sentence"] = batch["sentence"].replace("Ö","OE")
    batch["sentence"] = batch["sentence"].replace("Ü","UE")
    batch["sentence"] = batch["sentence"].replace("ß","SS")
    return batch

def predict(interpreter, audio):
    """Feed an audio signal with shape [1, len_signal] into the network and get the predictions"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Enable dynamic shape inputs
    interpreter.resize_tensor_input(input_details[0]["index"], audio.shape)
    interpreter.allocate_tensors()

    # Feed audio
    interpreter.set_tensor(input_details[0]["index"], audio)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data

def prediction_scorer(prediction, print_text=False):
    """Decode the network's prediction with an additional language model"""
    global beam_size, ds_alphabet, ds_scorer

    ldecoded = ctc_beam_search_decoder(
        prediction.tolist(),
        alphabet=ds_alphabet,
        beam_size=beam_size,
        cutoff_prob=1.0,
        cutoff_top_n=512,
        scorer=ds_scorer,
        hot_words=dict(),
        num_results=1,
    )
    lm_text = ldecoded[0][1]

    if print_text:
        print("Prediction scorer: {}".format(lm_text))
    return(lm_text)

def load_audio(wav_path):
    """Load wav file with the required format"""

    audio, sr = librosa.load(wav_path, sr=16_000, mono=True)

    len_audio = len(audio)
    audio = audio / np.iinfo(np.int16).max
    audio = np.expand_dims(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio, len_audio, sr


# ==================================================================================================

test_dataset = test_dataset.map(speech_file_to_array_fn)


#checkpoint_file = "./model/model_quantized.tflite"
checkpoint_file = "./model/model_full.tflite"
alphabet_path = "./data/de/alphabet.json"
ds_alphabet_path = "./data/de/alphabet.txt"
ds_scorer_path = "./model/kenlm_de_all.scorer"

beam_size = 256
labels_path = "./Scribosermo/extras/exporting/data/labels.json"

with open(alphabet_path, "r", encoding="utf-8") as file:
    alphabet = json.load(file)

with open(labels_path, "r", encoding="utf-8") as file:
    labels = json.load(file)

ds_alphabet = Alphabet(ds_alphabet_path)
ds_scorer = Scorer(
    alpha=0.931289039105002,
    beta=1.1834137581510284,
    scorer_path=ds_scorer_path,
    alphabet=ds_alphabet,
)


print("\nLoading model ...")
interpreter = tflite.Interpreter(
    model_path=checkpoint_file, num_threads=mp.cpu_count()
)

print("Input details:", interpreter.get_input_details())

print("\nRunning some initialization steps ...")
# Run some random predictions to initialize the model
for _ in range(5): #5
    st = time.time()
    length = random.randint(1234, 123456)
    data = np.random.uniform(-1, 1, [1, length]).astype(np.float32)
    _ = predict(interpreter, data)
    print("TM:", time.time() - st)

# Run random decoding steps to initialize the scorer
for _ in range(15): #15
    st = time.time()
    length = random.randint(123, 657)
    data = np.random.uniform(0, 1, [length, len(alphabet) + 1])
    prediction_scorer(data, print_text=False)
    print("TD:", time.time() - st)

print("\nInit done...")

predictions = []
total_inference_time = 0
total_input_duration = 0
for path in tqdm(test_dataset["path"]):
        audio, len_audio, sr = load_audio(path)
        input_duration = len_audio / sr
        ts_start = time.time()
        prediction = predict(interpreter, audio)
        predicted_text = prediction_scorer(prediction[0])
        ts_finish = time.time()
        predicted_text = re.sub(chars_to_ignore_regex, "", predicted_text).upper()
        predictions.append(predicted_text)
        total_input_duration += input_duration
        total_inference_time += (ts_finish-ts_start)    

references = [x.upper() for x in test_dataset["sentence"]]

## print all differences
# for i in range(len(predictions)):
#     if predictions[i] != references[i]:
#         print("prediction:", predictions[i])
#         print("reference :", references[i])
#         print("---")

print("Checkpoint file:", checkpoint_file)
print(f"WER: {wer.compute(predictions=predictions, references=references) * 100}")
print(f"CER: {cer.compute(predictions=predictions, references=references) * 100}")

print("Total inference time:", total_inference_time)
print("Total input duration:", total_input_duration)
print("Realtime factor:", total_inference_time / total_input_duration)