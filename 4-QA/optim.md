REDUCE 


### LLM principle
training time 
A research team is developing a new large-scale generative model and is encountering significant challenges with training time and memory consumption on their existing hardware. Which of the following strategies would best address these scalability and computational requirements?
- pruning and quantization to reduce model size
- mixed-precision training for memory efficiency

节流to reduce model size
Pruning（剪枝）- reduce components 
- 是指从神经网络中移除不重要的权重或神经元，以减少模型的复杂度。
- 权重剪枝（Weight Pruning）：移除权重值接近于零的连接。
- 结构剪枝（Structured Pruning）：移除整个神经元、通道或卷积核，更适合硬件加速。

Quantization（量化）- reduce density in each component
- 是将模型中的浮点数（如32位浮点）转换为更低精度的数值（如8位整数），以减少模型大小和加快计算速度。
- ost-training quantization（训练后量化）：在模型训练完成后进行量化。
- Quantization-aware training（量化感知训练）：在训练过程中考虑量化误差，提高精度。

开源
utilizing multiple GPUs for parallelized training


inference
Which of the following best reduces inference latency in a GenAI system?
- ONNX or TensorRT that optimize the execution graph and leverage hardware acceleration.
apply quantization (e.g., converting model weights from float32 to int8 or float16), which speeds up processing and reduces memory use.



### Nvidia tools
Deving LLM
What is NVIDIA NeMo(Neural Modules) primarily used for?
Developing and optimizing custom generative AI models
NeMo is part of NVIDIA AI Foundry, offering secure and scalable enterprise-grade AI development solutions.


serving/deploying for inference
What is NVIDIA NIM used for?
optimized inference microservices on NVIDIA-accelerated infrastructure


inerencing
TensorRT




What is the primary goal of model pruning in LLMs?
Model pruning removes less significant weights or neurons
reduce the model’s complexity, 
improving inference speed and 
reducing memory requirements without substantially affecting performance. 


How would you implement basic model pruning?
By identifying and removing weights or neurons that contribute little to the model’s output.


How to implement a Feedforward Neural Network(FFN) in an LLM to achieve optimal performance?
- Dropout reduces overfitting, while 
- layer normalization stabilizes and accelerates training.