# Local SMEL FT improvement

Upstream: https://github.com/artikulus1301/smel-ft

Pinned commit: `74a4635d5f7733baae9915d291573fae94a0499b`.

`smel-ft-74a4635-loss-validation.patch` adds boundary validation and four CPU
regression tests to the actual upstream frozen-head cross-entropy API. Invalid
targets, nonpositive/nonfinite denominators, invalid chunks and a trainable head
now fail explicitly. The valid-input algorithm is unchanged; its loss and scaled
hidden-state gradient are compared with full PyTorch CE.

Apply using `scripts/09_smel_setup.ps1`. It preserves an existing checkout whose
revision or modifications differ. The patch is local; it has not been submitted
to the upstream maintainer.

The upstream Apache-2.0 license and NOTICE are preserved here. Modified upstream
files carry a modification notice. Dense Qwen3 cache/LoRA support lives separately
in `izumxai/distill/smel_dense.py` and `train_smel.py`. It imports upstream's loss;
it does not implement or claim MoE routing for a dense model.

```powershell
$env:PYTHONPATH = "$PWD/tools/smel-ft"
.\.venv\Scripts\python.exe -m unittest discover -s tools/smel-ft/tests -p test_loss_validation.py -v
```
