"""Synthetic-data integration check; never assignment results. Run from this project directory."""
import json,tempfile,os
from pathlib import Path
from PIL import Image
import numpy as np
from udl.pipeline import config_signature,atomic_json
from udl.reporting import EXPERIMENTS,build_report,report_ready

def main():
    original=Path.cwd()
    cfg=json.loads(Path('config.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
     root=Path(tmp);cfg.update(run_dir=str(root/'runs'),smoke=True,device='cpu',train_examples=4,validation_examples=4,test_examples=4)
     (root/'runs').mkdir();atomic_json(root/'runs/environment.json',{'gpu':'synthetic unit-test fixture; NOT assignment results','cuda':None,'packages':{'torch':'test'}})
     (root/'observations.json').write_text(Path('observations.json').read_text())
     for name in EXPERIMENTS:
      path=root/'runs'/name;path.mkdir()
      row={'configuration_signature':config_signature(cfg),'experiment':name,'fid':None,'psnr':None if name.startswith('wgan') else 10.,
           'mse':None if name.startswith('wgan') else .1,'training_seconds':1.,'sampling_images_per_second':100.,'smoke_only':True,
           'codebook_perplexity':4.,'active_codes':4,'utilization':.01,'dead_codes':124,'prior_val_bits_per_code':2.}
      atomic_json(path/'metrics.json',row)
      history=[{'epoch':1,'val_psnr':10.,'utilization':.01,'wasserstein_estimate':.1,'generator_loss':.1}]
      atomic_json(path/'history.json',history)
      Image.fromarray(np.zeros((340,340,3),dtype=np.uint8)).save(path/'samples_100.png')
     os.chdir(root)
     pdf=build_report(cfg,final=False)
     assert pdf.is_file() and pdf.stat().st_size>1000
     try:report_ready(cfg,{})
     except ValueError as error:assert 'Smoke' in str(error)
     else:raise AssertionError('Smoke run was accepted as final')
     print('PASS: quantitative tables, figures, four 100-example pages and draft PDF render.')
     print('PASS: smoke results rejected for final submission.')
     os.chdir(original)

if __name__ == "__main__": main()
