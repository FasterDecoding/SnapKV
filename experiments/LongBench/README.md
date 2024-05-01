# Minimal version of LongBench
This is a VERY minimal implementation (with slightly modification) of LongBench.

You should be able to reproduce the results shown on our report.

Hardware: 1 * A100-80GB (2 for `Mixtral`)

example:

```
bash scripts/run_longbench.py
```

# Citation
```
@article{bai2023longbench,
  title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2308.14508},
  year={2023}
}
```