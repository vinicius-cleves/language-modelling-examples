# BERT

There is a lot of BERT implementations open sourced:

- The [BERT repository by Google Research](https://github.com/google-research/bert) provide a comprehensive step-by-step guide in order to train and finetune BERT. It, however, don't provide multi-gpu/multinode support. 
- [Nvidia's DeepLearningExamples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) provide a impresive scalable implementation of BERT, but it focus on reproducing BERT's training, and, in order to train with your own texts, you have to dive deep into the source code. 
- [Hugginface's transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) provides a masked language modelling training script that works out-of-the-box and scales to multi-gpu/multi-node. It, however, don't train with the Next Sentence Prediction objective necessary to train BERT.

We provide here an implementation that works out-of-the-box and is also scalable by adapting scripts from Google Research and Hugginface. As a bonus, our use of huggiface's transformers allows to use different pretrained BERT models as start point for the training. 