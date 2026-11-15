"""Synthetic-data integration check; never assignment results. Run from this project directory."""
import json, tempfile, os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from udl.pipeline import train_autoencoder,train_prior,train_gan,evaluate_generator
from udl.models import VQVAE,BetaVAE
from udl.reporting import metrics_table,build_report


def main():
    torch.set_num_threads(2)
    cfg=json.loads(Path('config.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
     cfg.update(run_dir=tmp,device='cpu',smoke=True,batch_size=2,evaluation_batch_size=2,workers=0,
                vq_epochs=1,prior_epochs=1,gan_epochs=1,n_critic=1)
     data=TensorDataset(torch.rand(4,3,32,32),torch.zeros(4))
     model,_,_=train_autoencoder(VQVAE(16,8),'vqvae_k16',data,data,cfg)
     prior,h,_=train_prior(model,'vqvae_k16',data,data,cfg)
     with torch.no_grad(): images=model.decode_codes(prior.sample(2))
     assert images.shape==(2,3,32,32)
     print('PASS: VQ-VAE training -> maps -> PixelCNN training -> ancestral sampling -> decoder.')
     for kind in ['wgan','wgan_gp']:
      generator,history,_=train_gan(kind,data,cfg)
      assert history[0]['critic_updates_cumulative']==2
      assert history[0]['generator_updates_cumulative']==2
      metrics=evaluate_generator(generator.sample,data,cfg,Path(tmp)/kind)
      samples=torch.load(Path(tmp)/kind/'samples_100.pt',weights_only=True)
      assert samples.shape==(100,3,32,32) and metrics['fid'] is None
      print('PASS:',kind,'actual optimizer steps, losses and 100 sample outputs.')
     from torchmetrics.image.fid import FrechetInceptionDistance
     class Features(torch.nn.Module):
      num_features=3
      def forward(self,x):return x.mean(dim=(2,3))
     metric=FrechetInceptionDistance(feature=Features(),normalize=False)
     x=torch.rand(30,3,32,32);metric.update(x,real=True);metric.update(x.clone(),real=False)
     assert abs(float(metric.compute()))<1e-5
     print('PASS: installed FID metric returns zero for identical feature distributions.')

if __name__ == "__main__": main()
