# generativetextmodel
COMPANY: CODTECH IT SOLUTIONS

NAME: B.ASHWINI

INTERN ID:CT2MTDM1406

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH


## 📄 Project Description: Generative Text Model using LSTM in TensorFlow

The “Generative Text Model using LSTM” is a deep learning-based natural language processing (NLP) project that focuses on creating a model capable of learning from a given corpus of text and generating new, coherent sequences based on that input. This project is implemented using Python and TensorFlow's Keras API and demonstrates the power of sequence modeling through recurrent neural networks (RNN), particularly using Long Short-Term Memory (LSTM) layers.

Text generation is a fundamental task in NLP that serves as the basis for many advanced applications, such as chatbots, predictive typing, AI-based storytelling, and intelligent assistants. In this mini-project, we developed a character-level and word-level predictive model that can predict the next word in a sequence based on previous context, and use this to iteratively generate complete sentences or phrases.



### 📘 Workflow Summary

1. **Data Preparation**:
   We started with a manually created small corpus consisting of simple, educational text related to AI, machine learning, deep learning, and NLP. The data was tokenized using Keras’ `Tokenizer` to convert words into numerical representations (word indices). From this, n-gram sequences were generated to enable the model to learn various phrase-level patterns from the input.

2. **Sequence Padding**:
   Since input sequences vary in length, we padded them to ensure uniformity using Keras' `pad_sequences`. This is essential because LSTM models require inputs of fixed dimensions.

3. **Feature and Label Splitting**:
   From each sequence, the last word was extracted as the target label, and the rest were considered as features. The labels were one-hot encoded using `to_categorical` so that the model can treat the output as a multi-class classification problem over the vocabulary.

4. **Model Architecture**:
   The model was constructed using a `Sequential` structure in Keras. It includes:

   * An `Embedding` layer to learn dense representations of words.
   * A single `LSTM` layer with 100 units to capture temporal dependencies.
   * A final `Dense` layer with a softmax activation function to output the next word's probability across the vocabulary.

5. **Training**:
   The model was trained over 200 epochs using the Adam optimizer and categorical crossentropy loss. The training logs show that the model gradually improved its loss and accuracy, indicating it successfully learned from the data.

6. **Text Generation**:
   A custom function `generate_text()` was used to generate new text sequences. The function takes a seed text and iteratively predicts the next word, appending it to the original seed. The final output, such as:

   > “Machine learning learning is simulation of human”
   > demonstrates that the model is capable of producing coherent, domain-relevant sequences, even though the training data was limited in size.



### 💡 Key Features

* Uses **TensorFlow and Keras** for deep learning.
* Implements **tokenization**, **n-gram generation**, and **padding** for sequence modeling.
* Employs an **LSTM-based architecture** ideal for language prediction.
* Capable of **generating text** dynamically from a seed phrase.
* Simple and modular code structure suitable for further extension.



### 🧠 Outcome & Learning

The project is a strong proof-of-concept for beginners in NLP and deep learning. Despite using a limited dataset, the model was able to generalize and generate contextually relevant phrases. It demonstrates how models can “learn” patterns in language and generate new content based on learned structures. This lays the foundation for building more complex applications such as chatbots, smart email composers, or AI writing assistants.


