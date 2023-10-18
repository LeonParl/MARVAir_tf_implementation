<h1 align="center">MARVAir TF Implementation</h1>
<h2 align="center">MARVAir: Meteorology Augmented Residual-Based Visual Approach for Crowdsourcing Air Quality Inference</h2>

> unlike many works that use PyTorch as their backend, this project is implemented using TensorFlow 2.x. <br>
> the contributor of the code: 
> [Muyan Yao](https://github.com/LeonParl) 
-----

Our implementation of MARVAir: Meteorology Augmented Residual-Based Visual Approach for Crowdsourcing Air Quality Inference. 

- This repository includes the implementation of the deep learning backbone network involved to extract features from the pre-processed user data. 
- Should you have any concerns, feel free contact with me directly at muyanyao \at ieee.org

If you use MARVAir in your project or research, fully or partially, please cite the following paper:
* [Muyan Yao, et al. (2022)](https://www.researchgate.net/publication/362169834)

```

> M. Yao, D. Tao, J. Wang, R. Gao and K. Sun, <br/>
> "MARVAir: Meteorology Augmented Residual-Based Visual Approach for Crowdsourcing Air Quality Inference,"  <br/>
> in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-10, 2022, Art no. 2514310, doi: 10.1109/TIM.2022.3193197. <br/>

```bibtex
@ARTICLE{9837081,
  author={Yao, Muyan and Tao, Dan and Wang, Jiangtao and Gao, Ruipeng and Sun, Kunning},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={MARVAir: Meteorology Augmented Residual-Based Visual Approach for Crowdsourcing Air Quality Inference}, 
  year={2022},
  volume={71},
  number={},
  pages={1-10},
  doi={10.1109/TIM.2022.3193197}}



```
## How to Install

The following dependency is required to have this project working normally:

- conda (anaconda, miniconda, or other variants)
- CUDA (if GPU based acceleration is preferred)

The python environment required for this project can be easily installed through: 
```bash
conda env create -f marvair.yaml
```

due to copyright concerns, the package of opencv has not been included in the environment description file. 
please install it manually before you dig into the code. 

Have a nice day! 










