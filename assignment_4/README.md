Final validation accuracy for this model within 20 epochs :- 99.52 % 

Maximum validation accuracy for this model within 20 epochs :- 99.57 % 

Number of parameters in the model :- 19,298


The general idea behind the design of this model was to gradually expand the number of kernels with each layer upto a maximum of 32 kernels in the middle, and then to gradually trim the number down to 10 for the final `conv` layer. Each `conv` layer is followed by a `BatchNorm` and `dropout` layer. I've added two `MaxPool` layers, first after the first four `conv` layers, and one before the last two `conv` layers. 

I also used a custom learning rate scheduler which reduced the learning rate after every epoch in order to allow the model to be trained at lower learning rates during the later epochs. This approach helped me improve my valid accuracy score from hovering around 99.43% to cross 99.5% . The validation loss also keeps going down throughout the training, which indicates that model performance might improve if trained further. 

The model attains a validation accuracy of 99.41 % in the 8th epoch and remains above 99.4 for the rest of the training. It achieves the highest validation accuracy of 99.57% in the 15th epoch. The final average validation loss stood at 0.0147. 


I experimented with a few transforms (`RandomRotation()`, `RandomHorizontalFlip()`, `RandomVerticalFlip()` etc), but got the best validation score with just the `ToTensor()` and `Normalize()`. I was also able to score 99.53 % validation accuracy with a smaller model containing 14,074 parameters.

