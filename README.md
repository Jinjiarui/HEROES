#Unbiased Ranking of Universal User Behavior
This is a *tensorflow* implement of our Model: ***Learning Universal User Behaviors with Hierarchical Recurrentsurvival Model for Unbiased Ranking*** and some
other baseline models(ESMM, ESMM2, DNN, MMOE).
##Dataset
We used two public datasets to verify the effect of our model: Alibaba Click and Conversion Prediction ([Ali-CCP](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408&userId=1)) and Criteo Attributions Modeling for Bidening Dataset([Criteo](http://apex.sjtu.edu.cn/datasets/13)). 

For the Ali-CCP, we have done a certain sampling work. The original training set and test set have 4KW browsing samples. We took out the browsing sequence containing at least one conversion behavior (that is, the continuous browsing of a user in the time series) as the actual sample.

In ESMM and MMOE models, sequence prediction is not involved, so we convert the data to tfrecord format in advance to speed up training. For the specific conversion code, refer to the get_tfreord*.py code in the [ESMM](https://github.com/Jinjiarui/HEROES/tree/master/ESMM) folder.

##Installation and Running
TensorFlow(>=1.14) and dependant packages (e.g., numpy and sklearn) should be pre-installed before running the code.

After package installation and data have been prepared and you are in the correct model folder, you can simply run the code:

`python model_dataset.py`

Where the model is in [esmm, esmm2, mmoe, dnn, heroes], and the dataset is in [alicpp, criteo].

We have set default hyperparameters in the model implementation. So the parameter arguments are optional for running the code.