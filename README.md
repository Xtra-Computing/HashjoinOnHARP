# HashjoinOnHARP

## Prerequisites
* The gcc-4.8 or above
* The HARPv2 platform 

## Run the code

Do not forget to set the PATH of the dataset. 

```sh
$ cd ./
$ make  # make the host execution program
$ aoc ./device/shj.cl -o ./bin/shj.aocx  # make the FPGA execution program. It takes time.
$ cd ./bin
$ ./host
```

## Cite this work
If you use it in your paper, please cite our work ([full version](https://www.comp.nus.edu.sg/~hebs/pub/cidr20-join.pdf)).
```
@inproceedings{chen2019fly,
  title={On-The-Fly Parallel Data Shuffling for Graph Processing on OpenCL-based FPGAs},
  author={Chen, Xinyu and Bajaj, Ronak and Chen, Yao and He, Jiong and He, Bingsheng and Wong, Weng-Fai and Chen, Deming},
  booktitle={2019 29th International Conference on Field Programmable Logic and Applications (FPL)},
  pages={67--73},
  year={2019},
  organization={IEEE}
}
@article{chenfpga,
  title={Is FPGA Useful for Hash Joins?},
  author={Chen, Xinyu and Chen, Yao and Bajaj, Ronak and He, Jiong and He, Bingsheng and Wong, Weng-Fai and Chen, Deming},
  year={2020},
  booktitle={CIDR 2020: Conference on Innovative Data Systems Research},
}

```
### Related publications
* Xinyu Chen*, Ronak Bajaj^, Yao Chen, Jiong He, Bingsheng He, Weng-Fai Wong and Deming Chen. [On-The-Fly Parallel Data Shuffling for Graph Processing on OpenCL-based FPGAs](https://www.comp.nus.edu.sg/~hebs/pub/fpl19-graph.pdf). FPL, 2019.


## Related systems

* Graph systems on GPU: [G3](https://github.com/Xtra-Computing/G3) | [Medusa](https://github.com/Xtra-Computing/Medusa)
* Other Thunder-series systems in Xtra NUS: [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) | [ThunderSVM](https://github.com/Xtra-Computing/thundersvm)
