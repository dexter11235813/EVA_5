## MODEL 1

### Target :- 
    - Take the model submitted for assignment 4, and reduce the number of kernels in each layer such that the number of parameters in the model are less than 10k 
### Results :- 
    - Total number of parameters = 8162
    - Best Train and Validation accuracy = 98.93% (14th epoch) and 99.25% (13th epoch) respectively

### Analysis :- 
    - Naively reducing the number of kernels from the previous submission gives me a decent base to begin with. I'm repeating a bunch of 3 * 3 Convolution layers after the first layer
      which stack up the parameter count without increasing the number of kernels available for feature extraction. 
    - The validation accuracy does not improve consistently with the number of epochs. It peaks at 99.25% and then drops down again to 99.02 in the final epoch. The model performance can be finetuned with varying the optimizer and learning rates, which I will experiment with later. 
 

## MODEL 2 

### Target :
    - Bring the number of model parameters below 8k by reducing 3 * 3 kernels / adding MaxPool layers in the model. 
    - Play around with optimizers to see their performance against SGD.


### Result :- 
    - Final Parameter count - 7384
    - Maximum Training accuracy - 98.831 (14th epoch)
    - Maximum Validation Accuracy - 99.29 (13th epoch)

### Analysis :- 
    - Changing the number of Conv2d and Maxpool layers did not significantly impact training performance, while brining down the model parameter count to below 8k. 
    - Even after reducing the number of parameters, the maximum training accuracy is better than the one observed in the model in file `model_1.ipynb`.
    - The difference between the training and validation accuracy suggests underfitting, reducing/removing dropouts might improve model performance.



## MODEL 3


### Target:
    - Check if varying the dropout rates improve training and validation accuracy 
    - Hit the target validation accuracy (99.40 %) at least once.
    
### Result :
    - Maximum Training Accuracy :- 99.178% (15th epoch) 
    - Maximum Validation Accuracy :- 99.40 (15th epoch) 
    - Final Parameter Count :- 7384
    
### Analysis :
     - Reducing the dropout rates improved both the training and validation accuracy. 
     - The model validation accuracy jumps around towards the second half of the training. Reducing LR with epoch might help
       improve model performance.


## MODEL 4

### Target :
     - Play around with custom learning rate schedulers to improve model stability towards the later half of training.

### Results :
     - Model Parameters = 7384
     - Maximum Training Accuracy = 99.25 (13th Epoch)
     - Maximum Validation Accuracy = 99.49 (14th Epoch)

### Analysis :
     - With a custom rate scheduler, the model exhibits smoother training behaviour, hitting the target score of 99.40% in the 11th epoch, and remains above it for the rest of the training. 
     - The model accuracy keeps on improving throuhout the training, indicating that the performance might improve with 
       further training.
