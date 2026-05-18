# MNIST Neural Network From Scratch

#SENTIMENT OF THIS PROJECT:
- This project follows a tutorial I found on the internet specifically from "Samson Zhang" in YouTube and his video on Neural Networks.
- With the growth of AI, I wanted to learn how the basic building blocks of it worked, starting with neural networks.
- I spent an entire week reviewing linear algebra as well as watching a large sum of tutorials, and lectures online regarding how such networks function.


- This project was created to better understand the core foundations behind neural networks and modern AI systems. This program was developed from scratch and every major function was implemented manually.

These functions are:
- Forward propagation
- Backpropagation
- Gradient descent
- ReLU activation
- Softmax activation
- One-hot encoding
- Parameter updates

# MNIST Dataset 
- This neural network is trained on the MNIST handwritten digit dataset which contains 70,000 images which are 28 by 28 pixels each.

---

# Project Motivation/ Resources

AI has always been a field that fascinates me and I wanted to learn its core fundamental i.e. the neural networks. I figured I'd develop such networks from scratch since I learn best by applying the knowledge I learn.

Before building this project, I spent time reviewing:

- Linear Algebra
- Matrix multiplication
- Partial derivatives
- Calculus 1
- Calculus 2

This project follows concepts learned from online lectures, tutorials, and educational resources, especially content from Samson Zhang.

---

# Result of This Neural Network

- with a learning rate of 0.1 and 1000 iterations, our accuracy was on average 88.39%
- Initially the models accuracy increased rapidly, but as we progressed through more itterations, the rate our accuracy got better decreased. A small explanation to why the rate of increase dropped is because at first the model knows of NOTHING so the amount that needed to be changed was HUGE and as we train the model more with more iterations, LESS needs to be altered each time.

---

# Limitations of This Neural Network

- having only one hidden layer is a big limitation for this network.
- having only 10 neurons in the hidden layer is also a limitation.

## Overall Explanation of the Limitations
- if we have more layers and more neurons in the hidden layer, we have more parameterswe can tweak and change. The more parameters we have, the more room for change we have. Its like having a basic car that turns roughly v.s a more complex car that turns more smoothly.

## How to Improve accuracy
- increase the number of hidden layers.
- increase the number of neurons per hidden layer.

---
# Neural Network Architecture

```text
Input Layer  -> Hidden Layer -> Output Layer
784 Neurons     10 Neurons      10 Neurons
