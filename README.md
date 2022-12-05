# FER
Facial Expression Recognition


# FER README

# 연구 배경

Facial Expression Recognition(FER)란 사람의 얼굴 이미지로부터 표정 혹은 감정을
인지하는 기술로, 최근 컴퓨터 비전 분야에서 그 중요성이 대두되고 있다. 현재 분류 체계로
쓰이는 건 1970년대 심리학자 Paul Ekman이 제안한 6개의 Basic Emotions(Happiness,
Sadness, Fear, Disgust, Anger, Surprise)에 무표정(Neutral)을 더한 7개 분류다. JAFFE,
KDEF 등의 데이터베이스에서 해당 분류를 사용하고 있다. 일부 연구에선 22개의 Compound
Facial Expressions를 사용한다. [6] FER은 CNN, RNN을 사용하지 않고 얼굴의 Feature를
기반으로 하여 추출한 벡터를 활용하는 방식으로도 오랫동안 연구됐다. [7]
FER의 고전적인 문제는 Label의 모호성이다. 사람이 표정을 분류하기에 주관적 인상이
반영되며, 인간의 표정이 가지고 있는 자체적인 복잡성 역시 영향을 끼쳐 이를 해결하려는
방법이 꾸준히 제안됐다.

## 관련 연구

Label의 모호성 문제를 해결하기 위해, Attention Heat Map의 일관성을 유지하는 방법과
Label 자체를 다루는 두 가지 방법이 사용되었다. Attention Heat Map이란 모델이 표정을
분류하는 근거로서 이미지 내의 중요성을 Heat Map 형태로 나타낸 이미지로, 블랙박스인
모델의 작동 원리를 파악하는 데에 이용된다. 이미지를 Flip 하거나 다른 변형을 가하더라도
일관성을 유지해야 한다는 가정으로부터, 이미지에 적용된 변환(크기 변화, 회전, 뒤집기
등)과 관계없이 이미지의 각 픽셀에 대한 Attention의 Consistency를 측정하는 방법이
제안되었다. [2, 5] 하나의 표정에 대해 하나의 분류가 정확하게 배정될 수 없으며, 분류자에
따라 하나의 표정이 다양한 분류로 나뉘는 문제를 해결하기 위해 하나의 표정에 여러
Label을 연결하는 Label Distribution Learning(LDL)이 제안됐다. [8, 3] 또한 모호성을
계산하여 가중치로 활용하거나 [1], 모호성이 큰 샘플을 삭제하는 방식이 사용된다.

## 연구 독창성

## 연구 내용

### 프로젝트 환경

데이터셋은 FER-2013으로, 48×48pixel의 Grayscale 이미지로 구성된다. 총 7개의 표정
분류로 BE를 따른다. 28,709개의 Training Set과 3,589개의 Test Set을 포함한다.

### word2Vec

단어 공간 사이의 거리는 Word2Vec등, 기존 연구에서 이미 계산되어 있다. Loss
function을 적용할 때 단어 사이의 거리(차이)를 고려하여 설계한다. 최종 Feature vector를
Ground truth로 지정한다.

**1. Layer**

```python
class CustomModel(keras.Model):
	def train_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			w = -100
		for n in range(len(weight) - 1) :
			a = cosine_similarity(a, y_pred)
		if w < a :
			w = a
			y_pred = n
		loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
		(...)
	return {m.name: m.result() for m in self.metrics}
```

**2. Loss Function**

```python
cosine_loss = tf.keras.losses.CosineSimilarity()
class NumLoss(tf.keras.losses.Loss) :
	def call(self, y_true, y_pred):
		y_true_n = y_true.numpy().tolist()[0]
		t = weight[y_true_n.index(int(max(y_true_n)))]
		y_pred_n = y_pred.numpy().tolist()[0]
		loss = tf.keras.losses.cosine_similarity(t, y_pred_n)
		for i in range(1, len(y_true.numpy().tolist())-2) :
			y_true_n = y_true.numpy().tolist()[i]
			t = weight[y_true_n.index(int(max(y_true_n)))]
			y_pred_n = y_pred.numpy().tolist()[i]
			loss += tf.keras.losses.cosine_similarity(t, y_pred_n)
	return tf.reduce_sum(loss)
```

```python
class NumLoss(tf.keras.losses.Loss) :
	def call(self, y_true, y_pred):
	y_true_n = y_true.numpy().tolist()[0]
	y_pred_n = y_pred.numpy().tolist()[0]
	difference_n=abs(cosine_loss(weight[int(max(y_true_n))], y_pred_n).numpy())
	for n in range(1, len(y_true.numpy().tolist())-2) :
		y_true_n = y_true.numpy().tolist()[n]
		y_pred_n = y_pred.numpy().tolist()[n]
		difference=abs(cosine_loss(weight[int(max(y_true_n))],
		y_pred_n).numpy())
		difference_n = difference + difference_n
	m = difference_n / _batch_size
	return m
```

**3. Word2Vec Implementation**

```python
from gensim.models.keyedvectors import KeyedVectors
word2VecModel =
KeyedVectors.load_word2vec_format('D:/gn/GoogleNews-vectors-negative300.bin',
binary=True)
angry = word2VecModel['angry']
(...)
surprise = word2VecModel['surprise']
weight = [angry, disgust, fear, happy, neutral, sad, surprise]
```

![Untitled]()

**4. Accuracy**

```python
def category_accuracy():
	def recall(y_true, y_pred):
		y_true_n = y_true.numpy().tolist()
		y_pred_n = y_pred.numpy().tolist()
		w = -100
		score = 0
		for n in range(1, len(y_true.numpy().tolist())-2) :
			a = abs(cosine_loss(weight[int(y_true_n[n][0])], y_pred_n).numpy())
			if w < a :
				w = a
				y_pred = n
			if n == y_true[n][0] :
				score += 1
	return score / _batch_size
return recall
```

### 참고 문헌

[1] Zhang, Y., Wang, C., & Deng, W. (2021). Relative Uncertainty Learning for
Facial Expression Recognition.
Advances in Neural Information Processing
Systems, 34, 17616-17627.
[2] Zhang, Y., Wang, C., Ling, X., & Deng, W. (2022). Learn From All: Erasing
Attention Consistency for Noisy Label Facial Expression Recognition.
arXiv
preprint arXiv:2207.10299.
[3] Chen, S., Wang, J., Chen, Y., Shi, Z., Geng, X., & Rui, Y. (2020). Label
distribution learning on auxiliary label space graphs for facial expression
recognition. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition (pp. 13984-13993).
[4] Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2018).
Autoaugment: Learning augmentation policies from data.
arXiv preprint
arXiv:1805.09501.
[5] Guo, H., Zheng, K., Fan, X., Yu, H., & Wang, S. (2019). Visual attention
consistency under image transforms for multi-label image classification. In
Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition (pp. 729-739).
1.
[6] Du, S., Tao, Y., & Martinez, A. M. (2014). Compound facial expressions of
emotion.
Proceedings of the national academy of sciences, 111(15),
E1454-E1462.
[7] Ghimire D, Lee J. Geometric Feature-Based Facial Expression Recognition in
Image Sequences Using Multi-Class AdaBoost and Support Vector Machines.
Sensors. 2013; 13(6):7714-7734. https://doi.org/10.3390/s130607714
[8] Geng, X. (2016). Label distribution learning.
IEEE Transactions on Knowledge
and Data Engineering, 28(7), 1734-1748.
[9] Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). Cutmix:
Regularization strategy to train strong classifiers with localizable features. In
Proceedings of the IEEE/CVF international conference on computer vision
(pp. 6023-6032)
