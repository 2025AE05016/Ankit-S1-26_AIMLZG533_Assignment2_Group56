# Local implementation validation

These checks use random synthetic tensors, not full CIFAR-10 training. Their numbers are not assignment results. No BITS lab session, official 2048-feature FID run or screenshot was produced here.

## test_implementation.py

```text
unit_vae 1/1: loss=256.9312, val PSNR=10.78 dB, 0.1s
Resuming /var/folders/13/7xjm05l53z3998d585m1ddsm0000gn/T/tmpth6wzq0i/unit_vae/last.pt after epoch 1
Resuming /var/folders/13/7xjm05l53z3998d585m1ddsm0000gn/T/tmpoqoodx4z/last.pt after epoch 3
test_actual_small_autoencoder_training_and_resume (__main__.ModelTests.test_actual_small_autoencoder_training_and_resume) ... ok
test_checkpoint_restores_model_optimizer_and_rng (__main__.ModelTests.test_checkpoint_restores_model_optimizer_and_rng) ... ok
test_gan_shapes_and_unbounded_critic (__main__.ModelTests.test_gan_shapes_and_unbounded_critic) ... ok
test_gradient_penalty_uses_per_image_gradient_norm (__main__.ModelTests.test_gradient_penalty_uses_per_image_gradient_norm) ... ok
test_pixelcnn_has_no_current_or_future_leakage (__main__.ModelTests.test_pixelcnn_has_no_current_or_future_leakage) ... ok
test_pixelcnn_sampling_and_training (__main__.ModelTests.test_pixelcnn_sampling_and_training) ... ok
test_psnr_range_one (__main__.ModelTests.test_psnr_range_one) ... ok
test_quantizer_codes_and_gradients (__main__.ModelTests.test_quantizer_codes_and_gradients) ... ok
test_vae_shape_range_and_objective (__main__.ModelTests.test_vae_shape_range_and_objective) ... ok
test_vae_zero_kl (__main__.ModelTests.test_vae_zero_kl) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.291s

OK

```
## test_pipeline.py

```text
vqvae_k16 1/1: loss=0.0902, val PSNR=10.78 dB, 0.1s
vqvae_k16 prior 1/1: val=1.686 bits/code
PASS: VQ-VAE training -> maps -> PixelCNN training -> ancestral sampling -> decoder.
wgan 1/1: D=-0.0002, G=-0.0000, GP=0.000, 0.1s
PASS: wgan actual optimizer steps, losses and 100 sample outputs.
wgan_gp 1/1: D=9.3967, G=0.0138, GP=0.941, 0.1s
PASS: wgan_gp actual optimizer steps, losses and 100 sample outputs.
PASS: installed FID metric returns zero for identical feature distributions.

```
## test_reporting.py

```text
Written: /var/folders/13/7xjm05l53z3998d585m1ddsm0000gn/T/tmphpeo1pz3/runs/PartC_DRAFT.pdf
PASS: quantitative tables, figures, four 100-example pages and draft PDF render.
PASS: smoke results rejected for final submission.

```

Both notebook structures and all code cells were also schema/syntax checked; HTML source previews were exported.
