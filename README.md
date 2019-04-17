# char-cnn-text-classification-keras

- Summarization: https://thisisiron.github.io/nlp/Char-level-CNN/
- PPT material: https://docs.google.com/presentation/d/1OpFnpL0BZkKadWoRvxkUbVPeHXSj8tpnpOchfNCv8hQ/edit?usp=sharing

## Requirements
- Python 3
- Tensorflow 1.12

## Training
```
python train.py
```

## Data
### Data Details
~~kaggle data: https://www.kaggle.com/c/word2vec-nlp-tutorial~~<br>
Sentiment140 - A Twitter Sentiment Analysis Tool: http://help.sentiment140.com/for-students/

### Data Download
[Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

## Experiment
|                       | paper CharCNN | improved CharCNN | CNN Sentence |
|-----------------------|---------------|------------------|--------------|
| Sentiment140 Test set | --%           | 82%              | --%          |

**CharCNN**
- optimizer: SGD
- alphabet: abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}
- embedding size: 69 (the number of alphabet)

**improved CharCNN**
- optimizer: Adam
- alphabet: all characters in a dataset
- embedding size: free

## TODO
- [X] Character Level CNN은 Dataset이 클 경우 좋은 결과를 보여주므로 Big Dataset으로 교체하여 실험
- [X] Jupyter code를 py code 형식으로 변환
- [ ] Save and Load 구현

## Reference
[Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)<br>
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
