# Running The Experiments

Pretty much just
```
python mup_paper_experiments/build_orchastrator.py --config_generator_file <file> --max_concurrent <concurrent>
```

For example
```
python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/depth_only.py --max_concurrent 30
```

Make sure you have wandb setup prior to runs