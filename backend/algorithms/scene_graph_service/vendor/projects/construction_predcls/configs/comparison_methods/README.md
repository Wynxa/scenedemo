# Construction Comparison Methods

This folder keeps paper-oriented comparison configs separate from the main task configs.

## Methods

- `construction_compare_penet.yaml`
  - standalone `PENET` baseline for `tools/relation_train_net.py`
- `construction_compare_vctree_htcl_opts.yaml`
  - `VCTreePredictor_HTCL` for `tools/HTCL_main.py`
- `construction_compare_motif_htcl_opts.yaml`
  - `MotifPredictor_HTCL` for `tools/HTCL_main.py`
- `construction_compare_transformer_htcl_opts.yaml`
  - `TransformerPredictor_HTCL` for `tools/HTCL_main.py`
- `construction_compare_penet_htcl_opts.yaml`
  - `PENET_HTCL` comparison run for `tools/HTCL_main.py`

## Notes

- All configs target the custom construction `predcls` split.
- Checkpoint/validation periods are stretched to the final stage so logs still report paper metrics, while intermediate checkpoint clutter is minimized.
- `HTCL_main.py` still writes the final artifacts required by its own stage transition logic. This is expected.

## Commands

`PENET`:

```bash
python tools/relation_train_net.py \
  --config-file projects/construction_predcls/configs/comparison_methods/construction_compare_penet.yaml
```

`VCTreePredictor_HTCL`:

```bash
python tools/HTCL_main.py \
  --config-file projects/construction_predcls/configs/construction_relation_base.yaml \
  --my_opts projects/construction_predcls/configs/comparison_methods/construction_compare_vctree_htcl_opts.yaml
```

`MotifPredictor_HTCL`:

```bash
python tools/HTCL_main.py \
  --config-file projects/construction_predcls/configs/construction_relation_base.yaml \
  --my_opts projects/construction_predcls/configs/comparison_methods/construction_compare_motif_htcl_opts.yaml
```

`TransformerPredictor_HTCL`:

```bash
python tools/HTCL_main.py \
  --config-file projects/construction_predcls/configs/construction_relation_base.yaml \
  --my_opts projects/construction_predcls/configs/comparison_methods/construction_compare_transformer_htcl_opts.yaml
```

`PENET_HTCL`:

```bash
python tools/HTCL_main.py \
  --config-file projects/construction_predcls/configs/construction_relation_base.yaml \
  --my_opts projects/construction_predcls/configs/comparison_methods/construction_compare_penet_htcl_opts.yaml
```
