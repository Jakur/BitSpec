# BitSpec

 **BitSpec** is a spectral augmentation method for automatic speech recognition (ASR) tasks.

 ## Installation

 1. Install [Pixi](<https://pixi.prefix.dev/latest/#installation>).
2. Clone this repository and navigate to the project directory.
3. Run the training script:

```
pixi run python train_torgo.py
```

 ## Usage

 ### Training

 BitSpec can be used to train Whisper models with spectral augmentation on the TORGO dataset using a **leave-one-speaker-out (LOSO)** evaluation setup.

 #### BitSpec augmentation

 To train a Whisper Base model with BitSpec augmentation, specify the speaker used for validation with `--loso_val_speaker`:

```
pixi run python train_torgo.py \
    --use_augmentation \
    --model_name openai/whisper-base \
    --loso_test_speaker none \
    --loso_val_speaker [loso] \
    --output_dir out
```

 The following speakers are available:

```
M01, M02, M03, M04, M05, F01, F03, F04
```

 Replace `[loso]` with one of the speaker IDs above.

 #### SpecAugment baseline

 To train a baseline model using **SpecAugment**:

```
pixi run python train_torgo.py \
    --spectral_aug specaugment \
    --model_name openai/whisper-base \
    --loso_test_speaker none \
    --loso_val_speaker [loso] \
    --output_dir specout
```

 ### Parameter-efficient fine-tuning with LoRA

 By default, the training configuration uses partial layer freezing, as described in the paper.

 Alternatively, you can use **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning:

```
pixi run python train_torgo.py \
    --use_augmentation \
    --model_name openai/whisper-small \
    --loso_test_speaker none \
    --loso_val_speaker [loso] \
    --output_dir outlora \
    --use_lora \
    --lora_alpha 32 \
    --lora_r 32
```

 ### Evaluation

 To evaluate a trained model, provide the model architecture, LOSO speaker, and checkpoint location:

```
pixi run python train_torgo.py \
    --model_name [architecture] \
    --loso_test_speaker none \
    --loso_val_speaker [loso] \
    --output_dir eval \
    --eval_only \
    --checkpoint [checkpoint]
```

 Replace:

 - `[architecture]` with the model architecture (e.g. `openai/whisper-base`)
- `[loso]` with the speaker ID to use for validation
- `[checkpoint]` with the path to the trained model checkpoint

 ## Example Workflow

 For example, to train and evaluate a BitSpec-augmented Whisper Base model using `M01` as the validation speaker:

 **Train:**

```
pixi run python train_torgo.py \
    --use_augmentation \
    --model_name openai/whisper-base \
    --loso_test_speaker none \
    --loso_val_speaker M01 \
    --output_dir out
```

 **Evaluate:**

```
pixi run python train_torgo.py \
    --model_name openai/whisper-base \
    --loso_test_speaker none \
    --loso_val_speaker M01 \
    --output_dir eval \
    --eval_only \
    --checkpoint out
```
