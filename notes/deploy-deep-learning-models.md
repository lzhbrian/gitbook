# Deploy Deep Learning Models

OK. Another FLAG.



#### 一些经验

* Profiling 看瓶颈
* NumPy、Pytorch等的一些写法改变就能带来很大的提升
  * copy、cast、矩阵乘法、初始化
* 推理引擎
  * 推理引擎自带的量化功能
* 直接把模型变小，看看精度会掉多少
* 拿 C++ 重写瓶颈部分 
* 知识蒸馏、其他高端的网络压缩方法



#### TensorRT

Quickstart Guide: [https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
)

![](../.gitbook/assets/image%20%284%29.png)

![](../.gitbook/assets/image%20%283%29.png)

* Install
  * 安装 onnx：conda install -c conda-forge onnx
  * 安装 pycuda：pip install pycuda
  * 安装 TensorRT 7.0.0 （[https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-install-guide/index.html\#installing-tar](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-install-guide/index.html)）
    * 安装 CUDA 10.0, CUDNN 7.6.5（只有这一步需要sudo，其实也可以把cuda装在自己的目录下，所以理论上来说，整体的安装其实可以不需要sudo）\([https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-support-matrix/index.html](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-support-matrix/index.html)\)
    * 下载 tensorrt tarball：[https://developer.nvidia.com/nvidia-tensorrt-7x-download](https://developer.nvidia.com/nvidia-tensorrt-7x-download)
    * export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:&lt;TensorRT-${version}/lib&gt;
    * pip install tensorrt-\*-cp3x-none-linux\_x86\_64.whl
    * pip install uff-0.6.9-py2.py3-none-any.whl
    * pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
* Pytorch-&gt;ONNX-&gt;TensorRT 
  * 编译 trtexec：
    * cd /data1/lzhgck/tensorrt/TensorRT-7.0.0.11/samples/trtexec 
    * CUDA\_INSTALL\_DIR=/usr/local/cuda-10.0 make 
    * 可执行文件会出现在： /data1/lzhgck/tensorrt/TensorRT-7.0.0.11/bin/trtexec 
    * 可以写入 ~/.bashrc： export PATH=$PATH:/data1/lzhgck/tensorrt/TensorRT-7.0.0.11/bin
  * [https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb](https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
    )
* Tensorflow-&gt;TF-TRT
  * [https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb    ](https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb
    )

#### 

