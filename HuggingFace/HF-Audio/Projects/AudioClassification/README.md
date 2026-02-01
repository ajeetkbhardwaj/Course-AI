# Audio Classification : Music Genre Classification

Note : This project based on learning from huggingface audio course

### 🎧 **Learn and Build (Audio Classification) Systems**

* **Audio classification** : assigning one or more **labels** to an **audio recording** based on its content.
* **Examples of audio classification tasks:**

  * Detecting **wake words** (e.g., “Hey Siri”).
  * Identifying **keywords** in speech (e.g., “temperature” in a spoken query).
  * Classifying **environmental sounds** (e.g., “car horn”, “siren”, “dog barking”).
* **Transformers** are widely used for **audio and speech processing** tasks.
* **Pre-trained audio transformers** can be adapted for different classification purposes.
* **Practical project we will:**

  * **Fine-tune a pre-trained transformer** model for **music genre classification** (e.g., pop, rock).
  * This approach underlies systems like  **Spotify’s recommendation algorithms** .
* **Objectives**

  * Identify and select **suitable pre-trained models** for audio classification.
  * Use the **🤗 Datasets library** and **Hugging Face Hub** to find audio datasets.
  * **Fine-tune** a model for  **genre classification** .
  * **Build a Gradio demo** to interactively classify your own songs.

### 1. **Pre-trained Models and Datasets for Audio Classification**

* The **Hugging Face Hub** hosts **500+ pre-trained models** for **audio classification**.
* Common audio classification tasks include **keyword spotting**, **language identification**, and **zero-shot classification**.
* The **`pipeline()`** class makes switching between models and tasks **fast and simple** — no code changes needed.

#### **Transformer Architectures for Audio Classification**

* **Goal:** Convert a sequence of audio inputs into a **single class label**.
* **Encoder-only models** are preferred because:
  * They map audio to hidden-state representations.
  * The mean of these representations → linear layer → class label output.
  * Example: BERT-style models for sequence classification.
* **Decoder-only** and **Encoder-decoder** models are **less efficient** and **slower**, since they generate multiple outputs.

**Installation**

* To ensure the latest audio pipeline features are available:

  ```bash
  pip install git+https://github.com/huggingface/transformers
  ```

##### Task-1 : **Keyword Spotting (KWS)**

* **Task:** Identify specific keywords in a spoken utterance.
* **Example Dataset:** `PolyAI/minds14` (intent-based speech queries in multiple languages).
* **Model Example:**
  * `"anton-l/xtreme_s_xlsr_300m_minds14"` → fine-tuned XLS-R model achieving ~90% accuracy.
* **Use Case:**
  * Classifying customer support calls (e.g., “pay_bill”, “card_issues”).
* **Code**
  ```python
  from datasets import load_dataset
  from transformers import pipeline

  minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
  classifier = pipeline("audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")
  classifier(minds[0]["audio"])
  ```

---

#### Task-2 : **Speech Commands**

* **Dataset:** `speech_commands` — 15 keyword classes + silence + unknown.
* Used for **simple voice commands** (“up”, “stop”, “yes”, etc.).
* **Model Example:** `"MIT/ast-finetuned-speech-commands-v2"` (Audio Spectrogram Transformer).
* **Use Case:**

  * On-device wake word detection (“Hey Siri”, “Hey Google”).
  * Lightweight models (~millions of parameters) for continuous listening.
* **Code**

  ```python
  from datasets import load_dataset
  from transformers import pipeline

  speech_commands = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)
  sample = next(iter(speech_commands))

  classifier = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
  classifier(sample["audio"].copy())
  ```

---

#### **We can exploring more models on Hugging Face Hub**

* Go to [https://huggingface.co/models](https://huggingface.co/models).
* Filter by:
  * **Task:** Audio Classification
  * **Dataset:** e.g., `speech_commands`, `minds14`, `fleurs`
* Quickly locate **fine-tuned models** for our dataset and task.

#### Task-3 : **Language Identification (LID)**

* **Task:** Identify the **language spoken** in an audio clip.
* **Dataset:** `google/fleurs` (102 languages, including low-resource ones).
* **Model Example:** `"sanchit-gandhi/whisper-medium-fleurs-lang-id"`.
* **Use Case:**

  * Automatically detect spoken language → choose appropriate ASR model.
* **Code**

  ```python
  fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
  sample = next(iter(fleurs))

  classifier = pipeline("audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id")
  classifier(sample["audio"])
  ```

---

#### Task-4 : **Zero-Shot Audio Classification**

* **Goal:** Classify **unseen classes** without re-training.
* **Model Type:** **CLAP (Contrastive Language-Audio Pre-training)**.

  * Takes **audio + text** as input.
  * Computes **similarity score** between them.
* **Dataset Example:** `ashraq/esc50` (Environmental sounds).
* **Use Case:** Classify arbitrary audio clips (e.g., “dog bark”, “vacuum cleaner”) without matching label sets.
* **Code**

  ```python
  from datasets import load_dataset
  from transformers import pipeline

  dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
  audio_sample = next(iter(dataset))["audio"]["array"]

  candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]

  classifier = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
  classifier(audio_sample, candidate_labels=candidate_labels)
  ```

---

#### **CLAP Model Limitations**

* CLAP excels at **environmental sound recognition**, not **speech or language** tasks.
* Works best when **candidate labels are descriptive phrases**.
* Less effective for fine-grained speech distinctions (e.g., between languages).

---

#### **Key Takeaways**

* **Encoder-only transformers** are best for **audio classification**.
* **Hugging Face Hub** simplifies model and dataset discovery.
* **`pipeline()`** allows **plug-and-play experimentation** with pre-trained models.
* **Zero-shot models (CLAP)** expand flexibility for unseen classes.
* Audio classification models vary by **use case**:

  * **KWS** → intent recognition.
  * **Speech Commands** → simple keyword detection.
  * **LID** → language recognition.
  * **Zero-shot** → general-purpose sound identification.

---

#### **Next Step**

* **Fine-tune a transformer model** for **music genre classification** using the same pipeline principles.
* Result: your own **music classifier** that predicts genres like “rock” or “pop.”

## References

1. https://huggingface.co/learn/audio-course/en/chapter4/classification_models
2.
