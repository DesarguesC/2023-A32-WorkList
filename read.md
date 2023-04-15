# 2023-A32-WorkList

## What's in 'utils.py'
GetDataset: to integrate input data and ouput data

load_array: input a integrated dataset with the format of (input, output) and you will get a data iterator

normalization: input a list to get (max number, min number, a standarized list), the list is reflected to [-1, 1]

ArrNorm: use function 'normalization' to standarize a np.ndarray type data

df2arr: transfer list to np.ndarray with dtype=np.float32

R_square: calculate R-squared criteria to evaluate input (predict, ground truth)


## Hyper-Parameters

First of all, when we are not training this model, we always fix **batch_size** as **1**.

When we apply this model onto a realistic stuation, setting a proper 'sequence length' is of vital importance.

Thus, now we propose a method to offer a guidance on adjusting 'sequence length'.

**Sequence Length** in our project stands not only for the time series length, but also reflect the environment condition on a certain district. Normally, we have $\textbf{Sequence Length} \in [3,30)$. 

There's no doubt that most of the time the value of the meteorological index changes little during the time in those districts where in other city the weather shows more stability, while that index hugely changes from day to day.

Based on these analysis above, when we apply this model into a city with relatively unstable weather condition, we should set **Sequence Length** to a smaller size, while under other circumstance, **Sequence Length** is set to a bigger one.

$\textbf{Sequence Length} \mapsto  \begin{cases} \in[3,15) \quad {stable\quad weather} \\  \in [15,30) \quad {unstable\quad weather} \end{cases}$ 

Additionally, we judge the stability of the weather in the specific area where the model is applied not from the climate characteristics that the area has been exhibiting, but from the weather conditions in the recent month in the area where the model is used (taking into account the contingency and suddenness of weather changes). Therefore, the specific setting of HP may be influenced by subjective judgment to some extent. To address this issue, we provide an alternative pretrained model and a comparison table of parameters. The weather data (temperature, wind direction, etc.) for the recent period is input, and our pretrained model will return a stability score, against which the table is checked to obtain the recommended HP settings we provide.




## 超参数

首先，当我们没有在训练模型时，我们总是固定 **batch_size**的值为**1**.

当我们将模型应用到实际场景中时，设置一个合适的时间序列长度**Sequence Length**来保持我们模型的泛化能力是十分重要的

因此，接下来我们提出一种调整**Sequence Length**的方法



事实上，我们方案种的**Sequence Length**不仅代表一个时间序列长度，另一方面也反映着当前地域的环境条件信息. 一般来说，我们有
$\qquad\qquad\qquad\qquad\quad\textbf{Sequence Length} \in [3,30)$. 

毫无疑问，大多数实践那些气候较为稳定地区的气候指标值随时间的变化较小，否则气候指标值可能会有较大变化。

根据上述分析，当在一个其后相对不稳定的地区使用模型时，需要设置一个更大的**Sequence Length**值来增加模型在当前条件下的泛化能力。而在一个气候相对稳定的地区，我们不太需要一个较大的**Sequence Length**值，这样只会增加计算开销。**Sequence Length**的取值可以总结如下：

$\textbf{Sequence Length} \mapsto  \begin{cases} \in[3,15) \quad {stable\quad weather} \\  \in [15,30) \quad {unstable\quad weather} \end{cases}$ 


另外，我们判断的应用此模型的具体地区的天气是否稳定时，不是从该地域一直以来表现出的气候特征来判断，而是根据使用模型地区近一个月的天气变化状况来判断（考虑到天气变化的偶然性和突发性）。因此，在具体设定**Sequence Length**时可能一定程度上受主观判断影响。针对这个问题，我们提供了另一个预训练模型（pretrained model）以及一份参数对照表。将近一段时间内的天气（气温、风向等）数据输入，我们的预训练模型将返回一个稳定性评分值，根据此分值查表获得我们提供的推荐**Sequence Length**设置。




