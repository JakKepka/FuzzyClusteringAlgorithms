# Fuzzy c-means-based incremental and dynamic model for time series classification

The repository presents the code and work conducted to investigate the Fuzzy c-means-based incremental and dynamic model for time series classification. Based on this research and its results, a scientific article was written, the abstract of which is presented below. We also provide the conclusions of the study.

### Abstract

This paper concerns the task of time series classification. In the discussed
setting, data appears in chunks, and we update a model after each incoming
chunk. Such a scenario is called incremental learning. In this paper, we
propose a novel incremental learning model built on the base of the Fuzzy
C-Means algorithm. As such, we contribute a new dynamic and incremental
version of the Fuzzy C-Means algorithm and complement it with additional
steps that result in a procedure allowing to conduct supervised learning that
enables time series classification. The new method is termed LDI-FCM,
short for Local Dynamic Fuzzy C-Means. In the paper, we demonstrate that
LDI-FCM works generally better than its baseline counterparts. First and
foremost, we show that it relieves the model designer from the challenging
and time-consuming task of hyperparameter tuning, which distinguishes LDI-
FCM from its competitors.

##### Keywords: incremental learning, dynamic clustering, fuzzy clustering, time series classification, voting

### Conclusion

In this paper, we have presented a new time series classification method
suitable for processing stream data. The new model, LDI-FCM, was designed
as an extension of the plain FCM model. It was equipped with two custom
voting procedures of similar practical usability.
The specific model construction process allows us to classify multivariate
time series data effectively. We have demonstrated that it performs better
than the plain FCM. The core challenge for non-dynamic clustering algo-
rithms like FCM is setting the optimal number of centroids. We show that
this number plays a significant role in this case. Thus, a risk can arise that
one will use an incorrect value. Our method eliminates this problem, as the
final number of centroids is determined dynamically while subsequent chunks
of temporal data appear.
Future development of this model may include incorporating an internal
step of chunk data analysis. We assume that each chunk belongs to a single
class. We may relax this assumption and revisit the chunk class assignment.


