# UDL Assignment 2 — Group 56

Prepared for **Ankit Kishor Bukshete — 2025AE05016**. Add the other members of your existing Assignment 1 group to `config.json`. Only one submission per group is required.

This is a new UDL project. The existing Distributed ML/FedAvg notebook and the previous RAG project are separate assignments and were not modified.

## What has been implemented

| Requirement | Implementation / evidence after the full run |
|---|---|
| CIFAR-10; pixels in [0,1] | Fixed 45,000/5,000/10,000 train/validation/test split; `ToTensor()` |
| Beta-VAE, beta = 1, 2, 4, 10 | All four models, same initialization/budget; per-epoch validation PSNR; final test PSNR/MSE and FID |
| VQ-VAE, K = 512, 256, 128 | All three models; reconstruction metrics and codebook histograms, perplexity, utilization, unused entries |
| Extract discrete latent maps | Saved `latent_maps.pt` for every K, with shapes and range displayed in Part A |
| Train PixelCNN prior and generate | Separate causal PixelCNN for each K; categorical likelihood; raster-order sampling; VQ decoder |
| WGAN | RMSprop, Wasserstein critic loss, critic weight clipping, random samples, final FID |
| WGAN-GP | Adam, per-image gradient penalty on interpolated real/fake pairs, no clipping, random samples, final FID |
| Part C: 100 examples per selected model | Four unselected 10×10 generated-image grids: VAE beta=1, VQ-VAE K=256, WGAN, WGAN-GP |
| Compare reconstruction, sampling quality and speed | Same test reference and FID counts, tables, curves, timed generation, training time, report generator |
| Notebooks + corresponding HTML | `UDL_Group56_PartA.ipynb` and `UDL_Group56_PartB.ipynb`; current HTML files are source previews |
| Part C PDF | `build_report` generates a draft after all runs; finalizer requires real results and your visual discussion |
| Mandatory BITS lab usage screenshot | You must capture and provide `BITS_lab_usage.jpg`; no screenshot is fabricated |

## 1. What you need to do first

1. Sign into **BITS Prayogshala for UDL S1-26_AIMLZG533** and allocate the provisioned GPU. Local/Colab training cannot replace the assignment's mandatory lab usage.
2. Upload and extract this entire project, including `udl/`, in a persistent working folder. Open a notebook from that folder and select the lab GPU kernel.
3. Keep the lab's working CUDA-compatible PyTorch/torchvision pair. Install missing dependencies using `%pip install -r requirements.txt` in a notebook cell; restart the kernel afterward if packages changed. The requirements preserve a working torch installation if it already satisfies them.
4. Edit `config.json`: add all group members, set `lab_confirmed` to `true` only after confirming the lab environment, and use `device: "cuda"`. The notebook prints and records the actual GPU.
5. Ensure the runtime can download CIFAR-10 and official torch-fidelity Inception weights. If the lab blocks outbound downloads, stage the official CIFAR-10 files under `data/` and the Inception checkpoint in the normal PyTorch cache before running. Do not replace Inception with a different classifier.

## 2. Verify, then train

Optional local checks:

```bash
python test_implementation.py
python test_pipeline.py
python test_reporting.py
```

A tiny pipeline check can be run with `python run.py a --smoke` and `python run.py b --smoke`. Smoke results are isolated in `runs/smoke`, skip final FID, and are not eligible for final submission. Their epochs and scores are not evidence of adequate training.

For the required work, set `SMOKE = False` and run **Part A**, then **Part B** in Jupyter. The initial full budget is 100 epochs per autoencoder, 50 per PixelCNN and 100 reported GAN epochs. There are nine model settings plus three priors. This is substantial GPU work; its duration depends on the allocated GPU. Monitor the first epochs before estimating total runtime. Training is not complete just because a loop finishes: inspect PSNR/loss curves, code utilization, prior cross-entropy and image quality.

Checkpoints and optimizer/RNG states are saved every epoch. Rerun with the same configuration to resume; completed experiments are reused. Do not edit training configuration or code inside an existing run: choose a new `run_dir` for a changed experiment. Use persistent lab storage so a reset does not delete checkpoints.

The optional CLI is useful for recovery:

```bash
python run.py a
python run.py b
```

After CLI training, run the notebooks too. Their training calls will reuse matching completed results and display the actual tables/images; save the notebooks to embed those outputs.

## 3. Evaluation choices

- Validation PSNR is recorded every epoch. Autoencoders and GANs use the final scheduled checkpoint. PixelCNN sampling uses `prior_best.pt`, selected by validation cross-entropy; `prior_last.pt` preserves the latest training state for recovery. No checkpoint is selected using test data or test FID.
- PSNR uses range 1 and averages per-image dB values. VAE reconstructions use posterior means. GAN reconstruction PSNR/MSE is **N/A**, since unconditional samples have no paired target.
- All final FIDs use 10,000 official test images and 10,000 generated images, standard 2,048-dimensional Inception features, float32 feature extraction and float64 statistics. A first-use weight download is required.
- VAE FID uses standard-normal prior samples; VQ FID uses learned PixelCNN prior samples. Reconstruction FID and uniform random code sampling are not substituted.
- Codebook quality uses whole-split counts, not the average of minibatch perplexities.
- VQ sampling time includes PixelCNN's 64 sequential latent decisions and decoding. Training time includes PixelCNN training for VQ models; FID/validation are excluded from the training-time counter.
- A reported GAN epoch performs one generator step per training minibatch and five critic steps per generator step. It is not equivalent to one autoencoder data pass; cumulative update counts are recorded.
- One seed is a practical assignment comparison, not a statistical claim that one architecture always wins. Do not alter scores or select the best-looking 100 samples to force an expected ranking.

## 4. Complete your analysis

1. Inspect every learning curve, reconstruction grid, generated grid and review flag. If training is inadequate, document another run and use a new run directory. No metric threshold or epoch count guarantees satisfactory marks.
2. Replace the interpretation paragraphs in both notebooks with conclusions supported by your actual results.
3. Fill `observations.json`: visual assessments of the four required model grids and an overall discussion of reconstruction, sampling quality and speed. The workflow checks at least 30 words per model and 50 overall to prevent an empty final report; these are workflow checks, not additional rubric requirements.
4. Capture an actual screenshot showing the BITS lab workspace and running assignment/GPU allocation, save it as a genuine JPEG named `BITS_lab_usage.jpg`, and check it is readable. The software hardware log does not replace the required screenshot.
5. Save both notebooks with outputs. Generate/review a draft PDF through Part B or `python run.py report`.

## 5. Finalize and upload

From the project terminal:

```bash
python finalize_submission.py
```

The finalizer verifies all nine full-run metrics, matching configuration, 10k/10k FID counts, saved notebook output cells, your written observations and a real JPG. It exports both HTML files and creates:

- `runs/full/PartC_Comparative_Analysis.pdf`
- `UDL_Group56_Assignment2_Submission.zip`

The ZIP contains the two executed notebooks, both HTML exports, Part C PDF, mandatory JPG, supporting code/configuration and compact result artifacts. Large datasets/checkpoints and smoke/test artifacts are excluded. Review the PDF and archive yourself before one group member uploads them. If you change `run_dir`, the PDF is under that directory.

## What remains your responsibility

**Provide lab access and runtime; confirm the allocated GPU; add remaining group members; execute the full training; inspect and interpret actual results; write the perceptual observations; capture the real lab screenshot; save/export and upload the group deliverables.** No lab login, real screenshot or training result has been fabricated here.

The brief specifies no fixed epoch count or numerical FID/PSNR cutoff. It explicitly deducts marks for inadequate training and unsatisfactory metrics, so convergence and sample quality require review. There is no deadline specified in the supplied two-page brief; check your course portal.

## Training corrections

The GAN generator stays in training mode during both critic and generator updates; critic-only fake images are computed under `no_grad`. Evaluation mode is reserved for sample generation/evaluation. PixelCNN sampling restores the validation-best prior, with its selected epoch recorded in metrics. Start a new `run_dir` if you already trained with the earlier code: the source-integrity check deliberately prevents mixing implementations.
