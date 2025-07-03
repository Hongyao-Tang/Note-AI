---
author: "Hongyao Tang"
title: "2.1 [DL] Landscape"
date: "2025-06-30"
description: "Landscape for DL world from perceptron to LLM"
tags: [
    "DL",
]
ShowToc: true
weight: 3
---

## Quick refresher
|Input | Output|Child|Arch|Activation|Loss|
|-|-|-|-|-|-|
Non-sequential data|Binary classification || Multil-layer | Sigmoid |Binary cross entropy|
||Multiclass calssification|| **Normalization** is about around zero to avoid vanishing gradient| Relu/softmax| Categorical cross-entry |
||Multiclass classification - Image recognization|| CNN||
||Regression||**Regularization** is about dropping or penalty to avoid overfitting|Linear/no activation|MSE|
|Sequential data| Regression||RNN/LSTM|
||Multiclass classification - Text autocompletion (char-by-char) ||
||Text autocompletion (token-by-token)||RNN with embedding|
|||Translation|attention+RNN/ED <br>-> self-attenion/multi-head/Transformer-RNN|
||NLP - understand the context|| BERT|
||NLP - text generation||GPT|


---
## Full landscape
| <div style="width:150px">Problem type| <div style="width:50px">note</div> | <div style="width:150px">Network topology</div> | <div style="width:150px">Activation function</div> | <div style="width:150px">Loss function</div> | <div style="width:150px">Adjust weights</div> |
|-|-|-|-|-|-|
| **Binary classification - logical OR/AND/NAND functions** | - | Single perceptron ![alt text](images/single-perceptron.png) | Sign ![alt text](images/sign.png) | y != y_hat | x[i] is the magnitude<br> y is the direction <br> L_R = 0.1 <br> w[i] += y * L_R * x[i] |
| **Binary classification - logical XOR function** | - | <span style="background-color:#ffcccc">Perceptron cannot separate XOR with a single straight line </span> <br> Multi perceptron - Two layer network ![alt text](images/two-layer-net.png) |  <span style="background-color:#ffcccc">Sign contains discontinuity. Need continous functions to use gradient descent </span> <br> Defferentiable active function: tanh for hidden, logistic for output ![alt text](images/tanh.png) | <span style="background-color:#ffcccc"> Sum of errors: 1.multiple errors may cancel each other out. 2.sum depends on num of examples </span> <br>  Mean squared error ![alt text](images/mse.png) | Gradient  ![alt text](images/gradient.png)  derivative is the sensitivity: 1.slope is steep, derivate is large, step are large. 2.slope is positive, minus, move to left <br> w = w -  L_R * de/dw|
|**Binary classification - if a patient has a specific condition based on a num of input variables (From bellow optimized)**| - | logistic output unit ![alt text](images/logi-output.png) | Sigmoid|Binary Cross Entropy|-|
|**Multiclass classification - classify MNIST handwritten digits**|-| <span style="background-color:#ffcccc">Need a multiclass output </span> <br> ![alt text](images/mnist.png)| - Sign | - MSE | - SDG |
|(DL framework) | -| Higher level | Configuration | Configuration | Automatic |
| <span style="background-color:#ffcccc">Saturated neuron to Vanishing gradients</span> <br>  Mitigation | (Input) & Hidden layer | Input layer - Input normalization/standardization to control the range of input  ![alt text](images/input-norm.png) <br><br>  Hidden layer - Batch normalization ![alt text](images/batch-norm.png)<br><br> Weights in every layer - Weight initialization ![alt text](images/w-init.png) | <span style="background-color:#ffcccc">using tanh hidden activation function</span> <br>   Different activation function: relu ![alt text](images/relu.png)|-|<span style="background-color:#ffcccc">Gradient direction diverge</span> <br> BATCH_SIZE <br> <br> <span style="background-color:#ffcccc">Fixed lr vibrating when converging</span> <br> dynamic L_R of Adam|
|-|Output layer| glorot_uniform for sigmoid |sigmoid|When using sigmoid funcction in output layer, use binary cross-entry loss function ![alt text](images/bce.png)|-|
|<span style="background-color:#ffcccc">Some sigmoid may have same value</span> <br> |-|![alt text](images/cce.png) | softmax is mutual exclusive|Categorical cross-entry loss function|-|
|**Multiclass classification - classify CIFAR-10 objects/ImageNet DS 1000 objetcs**|-|<span style="background-color:#ffcccc">Image needs to extract spatial features</span> <br> Adding convolutional layers <br> CNN ![alt text](images/cnn.png)  <br><br>  <span style="background-color:#ffcccc">Heavy computing for large image</span> <br>Max-pooling by model.add(MaxPooling2D(pool_size=(2,2), stride=2)) <br><br> <span style="background-color:#ffcccc">Edge pixel not learned</span> <br> Padding by padding='same' <br><br> <span style="background-color:#ffcccc">Inefficient CNN</span> <br>Depth-wise separable convolutions; EffificentNet |- ConLayer is relu; - Output layer is softmax| - Categorical cross-entry loss function|-|
|(Use a pre-trained network/model e.g. ResNet-50)|-|Arch: <br> AlexNet<br>  VGGNet - building block<br>  GoogLeNet - Inception module<br>  ResNet - Skip-Connection|-|-|-|
|(Customize a pre-trained network/model)|-|Transfer learning - Replace the output layers and retrain<br>Fine-tuning model - Retrain directly the upper layers <br> Data augmentation  - Create more training data from extsing data|-|-|-|
|**Regression - predict a numeral value rather than a class/predict demand and price for an item**|-|<span style="background-color:#ffcccc">Need to output any number without restriction from activation</span> <br> ![alt text](images/regre.png) The output layer consists of a single neuron with a linear activation function|Linear activaiton/no activation function|MSE|-|
|<span style="background-color:#ffcccc">ALL PROBLEM - training error rise/raise at end, not good</span>|-|Deeper(adding more layers) and wider(adding more neurons) network|-|-|-|
|<span style="background-color:#ffcccc">ALL PROBLEM- overfitting (test error rise at end, not training error)</span> <br> Regularizaiton|-| Drop-out neurons by model.add(Dropout(0.3))|-|Wright decay adding a penalty term to the loss function ![alt text](images/w-decay.png)|eary stopping|
|**Sequential data - Regression - Predict book sales based on historical sales data** | -| <span style="background-color:#ffcccc"> Need depends on previous inputs and requires memory. Fully Connected Networks (FCNs) cannot capture temporal dependencies.<br> To handle variable-length sequences, FCNs can only process fixed-length inputs. </span> <br> Add a Recurrent layer ![alt text](images/r-layer.png) RNN ![alt text](images/rnn.png) model.add(SimpleRNN(128, activation='relu')) |-|BPTT Unroll automatically|-|
|<span style="background-color:#ffcccc">Long sequence network, Weight multplication leads to gadient vanishing or explore </span> |-| Use LSTM layer instead of RNN layer ![alt text](images/lstm.png) LSTM(64,input_shape=(10,8)),|
|**Sequential data - Multiclass classification - Text autocompletion (char-by-char)**|-|<span style="background-color:#ffcccc">How to generate a sequence step by step</span> <br> Autoregression <br><br> <span style="background-color:#ffcccc">How to avoid greedy selection during the generation process</span> <br> Beam size|
|**<u>Text autocompletion (token-by-token)</u><br>- Speech recognition<br>- Translation**| RNN | Neural Language Models (RNN but  autoregression token-by-token) with embedding layer ![alt text](images/rnn-emb.png)|
|**- <u>Translation</u>** | RNN/ED+attenion|<span style="background-color:#ffcccc">Tranlation involves two languages, need two LM</span> <br> Encoder-Decoder network ![alt text](images/ed.png) <br><br> <span style="background-color:#ffcccc">When processing long texts, easy to 'forget' earlier information</span> <br> Attention mechanism<br>- ALL - every timestep has all context![alt text](images/context.png) <br>- FOCUS - Dynanically focus on different parts of context at different time steps - Attention network ![alt text](images/attn.png) ![alt text](images/attn2.png)|
||Self-attenion without RNN|<span style="background-color:#ffcccc">one attention<br>serial and slow</span> <br> Self-attention layer<br>- Mutli attentions(forms an attention layer)   <br>- Replace RNN with FCL<br>- Self - Q and KV all from self, remove dependency  <br>No dependencies between words, in parellel and fast ![alt text](images/self-attn.png) <br><br> <span style="background-color:#ffcccc">Capture features in one dimension only</span> <br> multi-head self-attention layer: multi-head can capture differenent aspects of features for one input ![alt text](images/multi-head.png) <br><br> Transformer<br>- multi-head attention layer<br>- multi-head  self-attention layer<br>- Norm layer<br>- Mutli-encoder-decoder-modules ![alt text](images/trans.png) ![alt text](images/tran2.png)|
|**NLP - understand the context<br> - better at understanding context , mainly used for comprehension tasks such as classification and labeling. <br> - but has weaker generation capabilities, needs to be combined with other models to perform text generation.**<br>● Sentiment analysis<br> ● Spam analysis<br> ● Classify second sentence as entailemnt, contradiction, or neutral<br> ● Identify words that answer a question|E|LLM -BERT <br> <u>Birdirectional</u> Predicting the middle part from the surrounding context — cloze-style tasks.  <br> <u>(Pretained)</u> as<br>● masked language model <br>● next-sentence prediction<br><u>Encoder Representations from Transformers</u> only ![alt text](images/bert.png)|
|**NLP - phrase the problem in a way that probability of a given completion can be interpreted as solution<br>Good at generating coherent and logically structured text, suitable for scenarios such as dialogue generation, article continuation, and creative writing** <br>● Sentiment analysis<br>● Entailment<br>● Similarity<br>● Multiple choice |D|LLM - GPT <br><u>Generative</u> predict next word <br><u>Pre-trained</u> as LM<br><u>Transformer</u> decoder only without cross-attention  ![alt text](images/gpt.png) ● GPT-2: Scaling the model make zero-shot much better <br>● GPT-3: providng in-context(learning at inference time) examples(few-shots) improves accuracy <br>- Codex/Copilot: Based on GPT-3, supervised fine-tuned (with code/docstring/ut as data) to generate Python code <br> - InstructGPT/ChatGPT: Based on GPT-3, SFT+RLHF, align model with user's intention <br>● GPT-4: Fine-tuned with RLHF, align model wtih Multi-Modal(text, image) Input