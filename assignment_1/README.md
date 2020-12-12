1.) Channel is a container of specific information. Only information that is contextually similar is represented in a channel. In case of images,  a channel encapsulates a specific kind of information about an image . A channel is a collection of same or similar features

A kernel is a matrix, which when convolved over a feature map, extracts a specific feature. 

for example, in an image, the Blue channel stores the intensity of the blue color, red channle stores the intensity of the red color, etc. 

1.) A channel is an encapsulation of a feature which encodes a specific kind of information about an image. A kernel is a matrix which extracts a specific feature from an image by convolving over a fixed channel.

2.) A 3 x 3 kernel is mostly used in feature extraction since it can be chained in a row to get the same Receptive Field as a larger kernel with fewer parameters. For example, using 2 3x3 kernels sequentially results in the same receptive field as a 5x5 , but uses fewer parameters( 18 vs 25).

3.) To reach 1x1 from 199x199, we would need a 100 3x3 convolution operation:-

199 x 199 > 197 x 197 > 195 x 195 > 193 x 193 > 191 x 191 > 189 x 189 > 187 x 187 > 185 x 185 > 183 x 183 > 181 x 181 > 179 x 179 > 177 x 177 > 175 x 175 > 173 x 173 > 171 x 171 > 169 x 169 > 167 x 167 > 165 x 165 > 163 x 163 > 161 x 161 > 159 x 159 > 157 x 157 > 155 x 155 > 153 x 153 > 151 x 151 > 149 x 149 > 147 x 147 > 145 x 145 > 143 x 143 > 141 x 141 > 139 x 139 > 137 x 137 > 135 x 135 > 133 x 133 > 131 x 131 > 129 x 129 > 127 x 127 > 125 x 125 > 123 x 123 > 121 x 121 > 119 x 119 > 117 x 117 > 115 x 115 > 113 x 113 > 111 x 111 > 109 x 109 > 107 x 107 > 105 x 105 > 103 x 103 > 101 x 101 > 99 x 99 > 97 x 97 > 95 x 95 > 93 x 93 > 91 x 91 > 89 x 89 > 87 x 87 > 85 x 85 > 83 x 83 > 81 x 81 > 79 x 79 > 77 x 77 > 75 x 75 > 73 x 73 > 71 x 71 > 69 x 69 > 67 x 67 > 65 x 65 > 63 x 63 > 61 x 61 > 59 x 59 > 57 x 57 > 55 x 55 > 53 x 53 > 51 x 51 > 49 x 49 > 47 x 47 > 45 x 45 > 43 x 43 > 41 x 41 > 39 x 39 > 37 x 37 > 35 x 35 > 33 x 33 > 31 x 31 > 29 x 29 > 27 x 27 > 25 x 25 > 23 x 23 > 21 x 21 > 19 x 19 > 17 x 17 > 15 x 15 > 13 x 13 > 11 x 11 > 9 x 9 > 7 x 7 > 5 x 5 > 3 x 3 > 1 x 1

4.) Kernels are initialized randomly are are then tuned using backpropagation. However, if the kernel values are initialized to a low value, and are multiplied by a small gradient, they tend to go towards zero, and thus no gradient is pass through them. Similarly, initializing kernel values to a large number leads to exploding gradients. To address this, the weights are initialized randomly , while ensuring that the mean of the weights of successive layers increases slowly and variance stays around 1.

5.) During initialization of a network, the weights of the network (kernel weights and bias for the CNN) are initialized randomly. In the forward pass, the input image is convolved over by the filters of the first layer to extract several features which represent a part of the image.Each layer of the model sees a larger part of the overall image and combines the lower level features to form a more abstract representation of an image. During training, the output of final layer of the model is compared with the expected output from the training set, and the difference between the two is calculated. 



This loss is then backpropagated to adjust the weights of the kernels based on their contribution to the final prediction. Therefore, the training tunes the kernel weights through successive adjustments to improve their ability to extract a feature of an image.