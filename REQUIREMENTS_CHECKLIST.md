# Exact brief-to-code checklist

Source: `UDL_Assignment2_Problem_Statement.pdf`, two pages supplied by the user. This checklist describes requirements; it does not claim the unexecuted experiments are complete.

## Part A — beta-VAE (2 marks)

- [x] Implement beta-VAE and settings 1, 2, 4, 10.
- [x] Train-loop code and checkpoint recovery for every setting.
- [x] Validation PSNR recorded each epoch and plotted together; final test PSNR separately labelled.
- [x] Final unconditional sample FID for each beta using a common reference/count.
- [ ] Run full training in BITS, inspect convergence and save notebook outputs.

## Part A — VQ-VAE with PixelCNN (3 marks)

- [x] Implement K=512,256,128 with straight-through quantization and commitment loss.
- [x] Reconstruction PSNR/MSE, codebook usage, utilization, unused entries and perplexity.
- [x] Extract and save discrete maps from the frozen final VQ model.
- [x] Train a separate causal PixelCNN for each K; record validation bits/code.
- [x] Ancestral code sampling and VQ decoding; common final FID protocol.
- [ ] Run all three autoencoders and priors, inspect collapse/quality and save outputs.

## Part B — WGAN and WGAN-GP (3 marks)

- [x] WGAN with weight clipping and RMSprop.
- [x] WGAN-GP with differentiable input-gradient penalty and Adam; no weight clipping.
- [x] Multiple critic updates, detached fake images for critic steps and correct generator loss.
- [x] Sample grids, final FID, loss histories, update counts and training/sampling times.
- [ ] Train on BITS and inspect the actual outputs; save all notebook results.

## Part C — comparative analysis (2 marks)

- [x] Code creates 100 generated examples each for beta=1 VAE, K=256 VQ-VAE, WGAN and WGAN-GP.
- [x] PDF generator inserts all four grids plus all nine experiment metrics and curves.
- [x] Compare reconstruction where defined, sampling/FID, codebook quality and speed.
- [ ] Write your actual perceptual observations and overall discussion; inspect the PDF.

## Submission and lab evidence

- [x] Two notebook implementations and corresponding source-preview HTML files prepared.
- [x] Full-run results and saved output cells checked by finalization script.
- [ ] Use mandatory Prayogshala UDL infrastructure and capture a real, readable JPG showing usage.
- [ ] Add all members from the existing Assignment 1 group.
- [ ] Save both executed notebooks, export final HTML and generate Part C final PDF.
- [ ] Review and upload once for Group 56.

The brief warns of deductions for inadequate training and unsatisfactory metrics but does not provide a numeric threshold or minimum number of epochs. Code completion is not proof of adequate model quality. No training results, sample-quality conclusions, or lab screenshots have been invented.
