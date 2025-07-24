Design consideration

\Route query to model\api gw
What is a key consideration when scaling a GenAI app using large models?
- Always using the largest model can be inefficient and expensive.
- Using model routing allows you to balance cost, speed, and quality by selecting models based on query type
  - routing simpler queries to smaller models and complex ones to larger LLMs.

Which of the following is a common rule used to route a query to a smaller, faster model?
- Simple or short queries typically don’t need the depth or complexity of large models, so routing them to smaller models saves cost and improves speed.

In a model routing layer, what is the role of query analysis?
- Query analysis (e.g., via token count, intent classification, or keyword detection)
- classify the query to decide which model should handle it

Why is caching often used in a model routing system?
- For identical or similar queries, reuse previous outputs not llm choice. 当某个请求或类似请求已经被处理过，系统会缓存其结果，下次遇到相同或相似请求时，直接返回缓存结果，而不是重新调用模型。
- reducing the number of LLM calls and saving time and money.


Which deployment strategy best supports model routing in production?



\Scalability
When designing for scalability in a GenAI system, which factor is most crucial?
handling increasing user loads and data volumes.
A single powerful GPU 
- lacks the fault tolerance
- lacks horizontal scaling capabilities needed for large-scale production systems.
Distributed computing frameworks for high availability
 cloud-native services enable horizontal scaling, efficient resource management
 

What is the primary concern regarding 'model drift' in a deployed Generative AI system?
changes in real-world data distribution.
- 欺诈行为的模式发生变化
- 用户行为模式改变
Degradation of model performance over time 