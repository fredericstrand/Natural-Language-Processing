# ChatGPT making jokes - from scratch

Before going into this section you should have an understanding of how Neural Networks works and some math/statistics intuition

This is a notebook aimed on giving you insight in how ChatGPT and other Large Language Models (LLM) actually work and how they are able to interperet and generate meaningful text.

In this example we will make a model generating jokes. To do this, we need to explore, among many things, the famous **transformers** and **attention**, outlined in the research paper "Attention Is All You Need". Link: https://arxiv.org/abs/1706.03762

Before you begin coding you should have a grasp of what these concepts mean

Language has a fundamental challenge: words depend on other words that might be far away in a sentence. Traditional neural networks processed text sequentially, often forgetting important context by the time they reached the end.

Transformers introduced the attention mechanism to solve this. Instead of processing words one by one, attention lets every word simultaneously examine all other words in the sequence. Think of it as each word asking: "What information do I need?" (queries), while other words respond with "Here's what I can offer" (keys) and "Here's my actual content" (values). Words with the strongest query-key matches get the most influence.

Multi-head attention runs several of these attention processes in parallel, allowing the model to focus on different types of relationships simultaneously - one head might track grammar while another follows meaning. This builds rich, context-aware word representations.

The complete transformer architecture wraps this attention system with feed-forward networks for processing, layer normalization for stability, and positional encoding to track word order. The key breakthrough: unlike sequential models, transformers can process entire sequences at once, making training massively more efficient and scalable.

## Requirements
- Numpy
- Pandas
- Tensorflow
- Scikit-Learn
