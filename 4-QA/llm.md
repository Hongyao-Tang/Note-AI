
## LLM





What is the main difference between the encoder and decoder in a transformer model?

- Encoders process input into context-rich representations, while decoders generate output tokens using the encoder’s output and previous tokens.

Encoder
- Input: Takes in the full input sequence (e.g., a sentence).
- How: Uses self-attention to understand relationships between all input tokens.
- Output: A sequence of contextual embeddings(alignment vector) that capture the meaning of the input.
Decoder
- Input: Takes the encoder’s output plus previously generated tokens.
- How: Uses cross-attention (on encoder output) and self-attention (on previous outputs).
- Output: A sequence of output tokens (e.g., translated sentence, summary, etc.).

What is the difference between self-attention and cross-attention in Transformer models?
- Self-attention focuses on one input sequence
- cross-attention links two sequences.


---













Which methods are commonly used to control randomness in LLM outputs, and how do they differ?
控制内容重复性
- Presence Penalty
  - penalizing tokens that have already appeared
  - Reduces the likelihood of repeated words to enhance variety in the output
- Frequency Penalty
  - avoid overuse of common words or phrases
  - 减少啰嗦
- Stop Sequence
  - stop the output when a specific sequence of tokens is generated
  - helping terminate outputs appropriately.

控制输出的多样性和随机性
- Top-p (Nucleus Sampling) 
  - Selects tokens from a fixed probability mass
  - coherence高质量: fixed
  - diversity不太常见的词: probability
- Top-k Sampling
  - Chooses tokens from a fixed number of top options
  - coherence
- Temperature/Noise
  - Adjusts the randomness in token selection
  - higher values diversity



How would you balance creativity and reliability in text generation using temperature in LLM?
- Adjust the temperature based on the context and desired output.
- value can be tuned depending on whether the task requires creativity (higher values) or reliability (lower values).









Lower-presecion













## SFT
How can you assess the success of fine-tuning a model?
- define performance metrics that are relevant to the specific task, such as for tasks like text generation.
  - accuracy
  - F1 score
  - BLEU score 


What is FT?
What role does 'fine-tuning' play in GenAI system design?
- a new, specific task or dataset
- training from scratch
- adaptation/fine-tuning - adjust the existing and pre-trained model to specialize



What is the key difference between Fine-Tuning and Transfer Learning?

- 迁移学习只更新“顶层”，新任务与原任务相似
- 微调可以更新“全身”,新任务与原任务差异较大



What is the main objective of PEFT in fine-tuning large language models?


- PEFT = Parameter-Efficient Fine-Tuning只微调模型中一小部分参数

| 方法名称       | 核心思想                                               | 优点                                       | 适用场景                     |
|----------------|--------------------------------------------------------|--------------------------------------------|------------------------------|
| **LoRA**       | 在权重矩阵中插入低秩矩阵，只训练插入部分              | 参数极少、效果好、易集成                   | 文本生成、分类、多任务学习   |
| **Adapter**    | 在 Transformer 层之间插入小模块，只训练这些模块        | 模块化强、可组合、适合多任务               | 多任务学习、迁移学习         |
| **Prefix Tuning** | 给每层注意力机制加上可训练的“前缀向量”              | 参数更少、适合大模型                       | 文本生成、对话系统           |

- reducing computational costs and energy consumption while maintaining model performance




What is the primary distinction between LoRA and QLoRA in the context of fine-tuning Large Language Models (LLMs)?
- fine-tune huge models, like a 70-billion-parameter one, on a single GPU
- LoRA adds low-rank matrices, reduce memory usage.
- QLoRA extends LoRA, by applying quantization(compresses the main LLM), reduce memory usage further.



What is meant by catastrophic forgetting during fine-tuning?
- while adapting to a new task, a pre-trained model loses previously learned knowledge 

What is a common cause of catastrophic forgetting in LLMs?
- model's weights are adjusted in a way that overwrites important information learned earlier


How would you mitigate catastrophic forgetting in LLMs during fine-tuning?
- regularization and continual learning help retain previously learned knowledge while fine-tuning, minimizing the risk of catastrophic forgetting.



## App
What is the primary purpose of the Runnable interface in LangChain?

Runnable 是 LangChain 中用于构建、组合和执行语言模型工作流workflow的标准接口
组件接口 - 所有实现了 Runnable 接口的组件都可以通过统一的方法调用，如：
- invoke()：处理单个输入
- batch()：并行处理多个输入
- stream()：流式输出结果
组合与链式调用 - 你可以使用 | 运算符将多个 Runnable 组件组合成一个“链”，实现复杂的工作流
- prompt | llm | output_parser
并发与异步执行
- Runnable 接口内建对并发执行的支持（如 batch_as_completed()），适合处理大量输入或 I/O 密集型任务。


What does the .bind() method on a Runnable do in LangChain?
- pre-configure or "lock in" certain input values 
- .bind() method on a Runnable
```py
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Define a prompt template with two variables
prompt = PromptTemplate.from_template("Translate '{text}' to {language}.")

# Bind the language to always be French
french_prompt = prompt.bind(language="French")

# Now you only need to provide 'text'
response = french_prompt.invoke({"text": "Hello, how are you?"})
```

What does the .with_retry() method on a Runnable do in LangChain?

- 临时性错误（transient failures）
  - 超时（Timeout）
  - 速率限制（Rate Limit）
  -  网络中断
- 错误通常不是你的代码有问题，而是外部服务暂时不可用

- .with_retry() adds retry logic自动重试失败的调用，避免整个链条因为一次小错误而崩溃。


拦截、记录、修改或响应模型生成过程中的事件
real-time monitoring and interaction with the LLM's workflow
How can you restrict user prompts to specific domains in LangChain?
- Before submitting to LLM, evaluate/validate the content of the prompt and determine whether it meets specific criteria, such as relevance to a domain.
- LangChain callback handler
What is a common use of StreamingStdOutCallbackHandler in Langchain?
- StdOutCallbackHandler - log input/output to a file
- StreamingStdOutCallbackHandler - prints each token as it’s generated
```py
from langchain.callbacks.base import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started with prompt:", prompts)

    def on_llm_new_token(self, token, **kwargs):
        print("New token generated:", token)

    def on_llm_end(self, response, **kwargs):
        print("LLM finished with response:", response)

# 使用 handler
from langchain.llms import OpenAI
llm = OpenAI(callbacks=[MyCustomHandler()])
llm("What is the capital of France?")
```



What is the primary purpose of LangGraph, a library built on LangChain?
How do LangFlow and LangGraph generally differ in their approach to development?
Which core functionality is a defining characteristic of LangFlow's interface?
What is the primary purpose of LangSmith in LangChain?


- Rapid prototyping
- LangFlow a low-code/no-code/a visual interface/a drag-and-drop interface

- Building avergae LLM applications
- LangChain a programmatic library

- Creating complex agents
  - stateful workflows: maintain state (memory of previous actions)
  - cyclical workflows: execute processes with loops or cycle
- LangGraph a programmatic library

- Testing and evaluating the behavior of language models (LLMs) and chains
- LangSmith 





## Prompt eng
How does context engineering differ from prompt engineering?
| 部分 | 说明 | 示例 |
|------|------|------|
| 1. 指令（Instruction） | 明确告诉模型你想让它做什么 | “请总结以下文章内容。”<br>“写一封道歉信。” |
| 2. 背景信息（Context） | 提供必要的背景或输入材料 | “这是一篇关于气候变化的文章：...” |
| 3. 角色设定（Persona/Role） | 指定模型的身份或语气 | “你是一个专业的历史老师。”<br>“用幽默的语气回答。” |
| 4. 输出格式（Format） | 指定你希望的输出形式 | “用表格列出。”<br>“输出为 JSON 格式。” |
| 5. 示例（Examples） | 给出你想要的风格或格式的例子（可选） | “例如：‘你好，张三。很高兴认识你。’” |
| 6. 限制条件（Constraints） | 限定输出的长度、风格、语言等 | “不超过100字。”<br>“用简体中文。” |

- Prompt engineering is about crafting instructions 
- Context engineering is about curating and structuring the right/relevant contexts — such as chat history, long-term memory, structured data, tool responses, and retrieved knowledge



What is a “PromptTemplate” in LangChain used for?
- Standardizing/Structuring prompts, with base template plus variables and context examples



How can you restrict user prompts to specific domains in LangChain?
- Before submitting to LLM, evaluate/validate the content of the prompt and determine whether it meets specific criteria, such as relevance to a domain.
- LangChain callback handler


Which of the following best describes the primary goal of Prompt Compression in the context of Large Language Models (LLMs)?
- make the input shorter without losing the critical/essential context needed for the LLM to generate a relevant response.


What is the main function of the "chain of thought" prompting technique?
-  break down its reasoning process into a step-by-step explanation
- mproves the quality of complex answers by encouraging logical and structured thinking.



What is the primary purpose of zero-shot and few-shot learning in LLMs?
- no or minimal examples provided in the prompt.


What is the fundamental difference between an 'open-ended prompt' and a 'closed prompt' when interacting with a generative AI model?
- open-ended prompts allow for broad, freeform generation
- closed prompts has constrains



How can you ensure that an LLM's output is in a certain format?
- ensure the output adheres to the required schema or structure, improving reliability and downstream processing
- Output parsers transform the raw output of an LLM into a structured/predefined format, such as JSON, XML, or tabular data. 





## RAG
What is a key advantage of Cache-Augmented Generation (CAG) over Retrieval-Augmented Generation (RAG)?
- RAG retrieves relevant documents at runtime
- CAG(Cache-Augmented Generation) preloads all necessary knowledge into the model before inference, eliminates retrieval latency


What is a major limitation of CAG compared to RAG?
- frequently updated knowledge such as news or stock market data.
- any updates to the knowledge base require re-caching the model’s context



What does HyDE stand for in the context of Generative AI?
- 直接用用户的问题去检索，可能太短、太模糊
- Hypothetical Document Embedding(HyDE) 用模型生成的“假设文档”去检索，可以更准确地表达用户意图
  - 用户输入问题（如：“什么是量子纠缠？”）
  - 语言模型生成一个“假设文档”，就像它自己写了一段关于这个问题的解释
  - 对这段假设文档进行嵌入（embedding）
  - 用这个嵌入去检索真实文档，找到更相关的内容



Index Ranking
- 相关文档找到了，但排得不好”的问题
- Index Ranking prioritize relevant documents higher in the results for a given query 确保最有用的信息排在最前面
  - Metrics like Precision@k, Recall, Mean Reciprocal Rank (MRR), and nDCG 


What issue does the "Lost-in-the-Middle" phenomenon highlight in Retrieval-Augmented Generation (RAG)?
- when LLMs process long contexts, they prioritize documents at the top and bottom while often ignoring those in the middle.
- This is why LangChain's LongContextReorder places the least relevant documents in the middle to enhance accuracy.




How does LangChain retain context across multiple user interactions in a conversational application?
- Without memory, each new LLM call is stateless—i.e., it knows nothing about previous turns.
- Across interactions, to retain conversational context 
- Memory module solves this by making the LLM appear stateful
1. Message Tracking:
- LangChain wraps each turn in the conversation as a message (e.g., HumanMessage, AIMessage).
- These are stored in a memory store such as
  - ConversationBufferMemory: entire history
  - ConversationSummaryMemory: summarized history
  - ConversationTokenBufferMemory: a sliding window of the most recent messages based on token limit.
1. Prompt Injection:
- When generating a new prompt for the LLM, LangChain automatically injects past messages from memory into the prompt, ensuring that the model has access to prior context.


Which concept aims to address the limitations of retrieving diverse and relevant information for large language models, often focusing on balancing relevance and novelty?
- retrieving diverse and relevant information for large language models
- Lord of the Retrievers (LOTR/Merger Retriever) - 将多个不同的文档检索器（retrievers）的结果合并成一个统一的结果列表，以提升检索的准确性和多样性。



What is a primary benefit of integrating knowledge graphs with Large Language Models (LLMs)?
- “幻觉hallucinations”指的是模型生成了看起来合理但实际上错误或虚构的信息。这在医疗、法律、金融等高风险领域尤其危险。
- Knowledge Graphs（知识图谱）知识图谱是一种图结构，节点表示实体（如人、地点、事件），边表示它们之间的关系。它们通常来源于高质量数据库（如 Wikidata、UMLS、DBpedia），因此具备高度的事实性和可验证性
- 通过事实验证（factual verification）来增强模型的输出可信度


relationships between SQL tables
You're building a Text-to-SQL system to convert user queries into SQL statements.
Your database is large and complex, with hundreds of tables and columns. You need to help the LLM understand relationships between these tables (e.g., foreign keys, hierarchies, and entity connections).
Which retrieval method would best support relationship-aware SQL generation?
- Use a Knowledge Graph, which captures the semantic and relational links between tables and columns in the schema.




How will you handle tabular, image, and text data present in a document for embedding generation in a RAG chatbot?
- Apply OCR or use an LLM model capable of directly processing images to extract and summarize text or content from image-based sections.





You are building a document processing pipeline for scanned PDFs containing a mix of paragraph text, tables, and embedded images. Your goal is to extract each data type separately for downstream processing — for example, summarizing text, structuring tables, and analyzing image content.
Which of the following is the most appropriate approach to handle this task?

unstructured library
- first separate document elements by type
- then apply specialized models (e.g., table parsers or vision models) to each type accordingly.



A financial research firm is developing an AI assistant to answer investor queries. The assistant must provide insights based on company financial reports, industry trends, and real-time stock prices.
Which approach, Cache-Augmented Generation (CAG) or Retrieval-Augmented Generation (RAG), would be the best fit for this AI assistant, and why?
Both (Hybrid)
- CAG caching long-term knowledge(financial reports)
- RAG dynamically retrieving real-time data(ndustry trends, and real-time stock prices) when necessary.


## Agent

What is the purpose of ReAct (Reasoning and Action) in LLMs?

ReAct (Reasoning and Action) is a framework that combines reasoning with external actions.


What is the primary role of the Tool Decoder in LangChain?
- Reading function docstrings 
   -  To recognize functions as tools
   -  help the LLM decide when and how to use them.


What is the main purpose of the A2A (Agent-to-Agent) protocol?
2A (Agent-to-Agent) protocol is common communication standard for Agents, regardless their
- architectures
- platforms 
- platforms 

- Each agent exposes a public Agent Card describing its capabilities (e.g., "I can optimize HVAC energy" or "I provide user preferences").
- Other agents can then:
  - discover -Discover these capabilities automatically
  - communicate - Send structured requests to delegate sub-tasks
  - collaborate - Negotiate formats for UX or data (e.g., text, forms, or JSON)



Which statement best describes the primary strength of CrewAI compared to LangChain agents in the context of multi-agent systems?

- LangChain agents offer more flexibility for single-agent tasks and broader integrations.
- CrewAI is generally better for complex, multi-agent workflows due to its opinionated structure for collaboration, 



What is AutoGen?

AutoGen is an open-source framework that helps developers create and manage AI agents that can work together to solve tasks.
AutoGen v0.4 introduces asynchronous messaging, which allows agents to communicate using both event-driven and request/response interaction patterns. This improves scalability and flexibility in multi-agent workflows.






## Eng
What does “observability” in LLMOps emphasize?
- understand behavior and root causes
- gathers detailed telemetry about prompts, output quality, latencies, and system traces


understanding complex internal model behavior,
- To gain deep insights into a Generative AI model's internal processing steps within a production application, which general logging principle is most effective?
- 
- simple input/output logging not enough
- Structured and contextual logging - provides granular details about how the model processes information, including intermediate transformations, choices made during generation, and interactions with external tools or data (like RAG)



## Safety
What does red teaming refer to in the context of Generative AI systems?
网络安全
- Red team 是模拟黑客攻击的团队，尝试入侵系统。
- Blue team 是防守方，负责检测和阻止攻击。
- 目标：测试组织的防御能力，找出安全漏洞。
AI safety/security evaluation
- stress-test GenAI models
  - uncover biases, security gaps, hallucination risks, and harmful outputs
- Red teaming involves adversarial tactics对抗性策略 (simulating real-world attacks or misuse scenarios)
- To test and improve model safety and reliability



How would you reduce the risk of adversarial examples affecting critical LLM deployments?
- By training the model on adversarial datasets to improve robustness



What is Gradient Leakage in the context of LLMs?
梯度泄露 指的是：攻击者通过访问模型的梯度信息，反推出训练数据的内容，甚至是原始输入样本。
换句话说，虽然你没有直接暴露数据，但你上传的梯度可能已经“泄露”了数据的秘密。

在训练过程中，模型根据输入数据计算损失函数的梯度，然后用这些梯度来更新参数。
攻击者如果能获取这些梯度，就可以通过优化算法反向推理出原始输入数据
A potential privacy risk where training data can be reconstructed from gradients



How would you define the purpose of privacy-preserving techniques in addressing gradient leakage?
- Privacy-preserving techniques like 
  - differential privacy - introduces noise into gradients, making it difficult to reverse-engineer sensitive training data while preserving model utility.
  - or gradient clipping are designed to prevent the leakage of sensitive data
  





What is a "jailbreak" in the context of LLMs?
- exploiting weaknesses in an LLM's prompt-handling mechanisms 
  - crafting malicious prompts to make the model produce unintended or unsafe outputs.
- to bypass restrictions or safety guidelines.


What is the primary purpose of guardrails in LLM applications?

safety measures control the model’s responses
- detect and prevent unwanted outputs, maintain ethical standards, and keep the application aligned with business or security policies



## GAN
生成式 AI 的不同分支
- BERT/GPT 是用于理解和生成自然语言
- GAN/VAE/Flow/Diffusion 是用于生成图像



VAE(Variational Autoencoder变分自编码器)
- Autoencoder 是什么？
  - Encoder：将输入压缩成一个低维表示（latent vector）
  - Decoder：从这个表示中重建原始输入
  - 但普通的 autoencoder 只是压缩和重建，不能生成新数据。
- Variational 是什么？
  - 在 VAE 中，我们不只是学习一个固定的向量表示，而是学习一个分布（通常是高斯分布）。
  - 编码器输出的是 均值 μ 和 方差 σ，表示潜在空间中的一个概率分布。
  - 然后我们从这个分布中采样一个向量 z，送入解码器生成数据。
  - 这就是“变分”的含义 —— 用近似分布来逼近真实的后验分布。
- 过程
  - 编码器（Encoder）：将输入数据压缩为潜在变量的分布（均值 μ 和方差 σ）
  - 采样器（Sampling）：从该分布中采样一个潜在向量 z
  - 解码器（Decoder）：将 z 解码为重建数据
- 生成图像质量通常不如 GAN（模糊）

GAN(Generative Adversarial Network生成对抗网络)
- 过程
  - 生成器（Generator）：从随机噪声中生成“假数据”
  - 判别器（Discriminator）：判断输入是真实数据还是生成数据
- 两者通过博弈训练，最终生成器能“骗过”判别器，生成高质量数据。
- 生成图像质量高、细节丰富
- 训练不稳定，容易崩溃（mode collapse）

Which statement best describes Diffusion Models, their role in Generative AI, and their key advantages?
- 传统的生成对抗网络（GAN）虽然能生成高质量图像，但训练过程容易不稳定，常出现模式崩溃（mode collapse）。
- Diffusion 的解决方式：通过逐步“去噪”的方式生成图像，训练过程更稳定，不依赖对抗机制。
  - 逐步扩散：将真实图像逐步加入噪声，直到变成纯噪声。
  - 逐步去噪：训练模型学习如何一步步从噪声中“去噪”，最终恢复出原始图像。

Stable Diffusion 是一种 Latent Diffusion Model
- Encoder（VAE）：将图像压缩为潜在向量 z
- Diffusion Process：在 z上进行加噪和去噪训练
- Decoder（VAE）：将去噪后的 z 解码为图像

What does the concept of 'latent space' refer to in Generative AI?
latent space潜在空间
- GAN 的潜在空间是压缩的：它把复杂的现实数据（如人脸图像）压缩成一个低维的向量表示。
  - A lower-dimensional representation of input data  是对数据的“抽象表达”，保留了最重要的特征
- 这个空间是有意义的：潜在空间中的每一个点，经过生成器映射后，都会生成一个具有特定特征的图像。
- 这些特征是可解释的：比如在 StyleGAN 中，移动潜在向量的某个方向可能会让人脸变老、变年轻、改变发型等。
  - used by generative models like GANs or VAEs to generate new, similar data.


A game developer is using a Generative Adversarial Network (GAN) to create diverse character portraits. They want to be able to smoothly transition between different facial expressions (e.g., happy to sad) or adjust specific features like hair color without retraining the entire model. Which concept should the developer leverage, and why is it crucial for achieving this control?
- Utilizing the latent space of the GAN, because points in this compressed space correspond to interpretable features, allowing for controlled manipulation and interpolation of generated attributes.





What is a key difference between discriminative and generative AI?
- Discriminative AI, such as sentiment classifiers, predicts labels using conditional probabilities. 
- Generative AI, like GPT, generate data




What is disentanglement in GANs, and why is it considered important?
GAN 的潜在向量z中的每个维度或子空间应该对应一个可解释的、独立的语义因素，比如：
一个维度控制“头发颜色”
一个维度控制“脸的朝向”
一个维度控制“是否戴眼镜”
这就叫做语义解耦（semantic disentanglement）。
disentanglement（解耦） 是指模型在潜在空间（latent space）中学到的不同维度能够独立控制生成样本的不同语义特征，而不是混杂在一起



What defines 'mode collapse' in GANs, and how does 'minibatch discrimination' address it?
- Mode collapse is when the generator produces limited output varieties, missing the training data's diversity
- Minibatch discrimination小批量判别 mitigates this by penalizing the generator for similar outputs within a batch, encouraging broader data distribution coverage.


损失函数/容易导致训练不稳定
How does the Wasserstein distance contribute to stabilizing GAN training, and what is its primary advantage over the standard minimization of JS Divergence?
- JS Divergence
  - with little distribution overlap, vanishing gradient
- Wasserstein distance
  - measures the 'earth mover's distance' between probability distributions
  - providing a more stable and meaningful gradient, leading to smoother optimization and better convergence.


WGANs (Wasserstein Generative Adversarial Networks)
- Using the Wasserstein distance (Earth Mover's Distance) as their loss function instead of traditional GAN losses. This distance provides smoother, more informative gradients to the generator, even when generated and real data distributions have little overlap.
- Employing a "critic" instead of a discriminator, which outputs a continuous score estimating the Wasserstein distance, offering a better signal for training.
- Often incorporating a gradient penalty (WGAN-GP) to enforce the Lipschitz constraint on the critic, further enhancing stability and performance without the issues of weight clipping.

Which of the following sets of techniques are effective in ensuring stability and convergence when training generative models, particularly GANs?

- spectral normalization
- gradient penalty (for WGANs)
-  label smoothing
-  


Which statement best describes CLIP (Contrastive Language-Image Pre-training), how it connects text and images, and its primary applications?
- zero-shot image classification, text-to-image generation guidance, and cross-modal search
- learn joint representations of text and images
-  CLIP (Contrastive Language-Image Pre-training)
   - a dual-encoder model that learns to map images and text into a shared embedding space by
   - contrastive loss对比学习 - maximizing the similarity(closer in space) of matching pairs and minimizing it(apart in space) for non-matching pairs