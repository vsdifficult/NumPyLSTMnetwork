# NumPyLSTMnetwork
This network writen only on numpy python lib
# How it's work? 
Long Short-Term Memory (LSTM), which is a type of recurrent neural network (RNN) capable of storing information for long periods of time. LSTM networks are widely used in the field of machine learning to solve problems of time series prediction, natural language processing, speech recognition and others. They are well suited for working with sequential data and have the ability to take into account dependencies between elements of the sequence. 

![изображение](https://github.com/vsdifficult/NumPyLSTMnetwork/assets/101355829/39ec6684-ceb4-4238-954e-bbb1a0da9f5f)

# Some math
```
ft = σ(Wf · [ht−1, xt] + bf ) 
it = σ(Wi · [ht−1, xt] + bi) 
˜ct = tanh(Wc · [ht−1, xt] + bc)  
ct = ft · ct−1 + it · ˜ct  
ot = σ(Wo [ht−1, xt] + bo)  
ht = ot · tanh(ct) 
```
