# http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
# http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/
# https://blog.naver.com/PostView.nhn?blogId=phj8498&logNo=221271057078
# https://blog.naver.com/PostView.nhn?blogId=phj8498&logNo=221271058287

import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# % matplotlib
# inline
plt.style.use('ggplot')


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        # start += (window_size / 2)
        start += (window_size // 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            # label = fn.split('/')[2].split('-')[1]
            label = fn.split('/')[2].split('_')[1]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)

                    # logspec = librosa.logamplitude(melspec)

                    # D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
                    # # https://librosa.github.io/librosa/changelog.html?highlight=logamplitude
                    # # D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)

                    logspec = librosa.core.amplitude_to_db(melspec)

                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# ===========================================

parent_dir = 'Sound-Data'
tr_sub_dirs= ['fold1','fold2']
tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
tr_labels = one_hot_encode(tr_labels)

ts_sub_dirs= ['fold3']
ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
ts_labels = one_hot_encode(ts_labels)

print ("middle")

# ========================================

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


# ==============================================


frames = 41
bands = 60

feature_size = 2460 #60x41
# num_labels = 10
# num_labels = 2
num_labels = 4
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01
total_iterations = 2000


# ==============================================


X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])
# Y = tf.placeholder(tf.float32, shape=(18,2))
# ValueError: Dimensions must be equal, but are 2 and 10 for 'mul' (op: 'Mul') with input shapes: [18,2], [?,10].
# 이런 오류가 뜸.
# 144 줄에서 loss에서 곱할때 오류가 뜨는걸 확인.
# 오류 해결!!!!!!! 위의 변수 지정할 때 내가 custom 해서 변수를 지정하는 것이었다.
# 그러므로 다른 코드들 다 원래대로 돌려놓고 하니 되긴 됨.
# fold1, fold2, fold3 에 무슨 wav를 넣어야 되는지 알아야 된다.

cov = apply_convolution(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
# y_ = Tensor("Softmax:0", shape=(?, 10), dtype=float32)
# f 는 (?,200), out_weights은 (200,10) 차워인데 num_hidden = 200, num_labels = 10 이다. 이 두 변수를 조절해서 (18,2)로 강제로 맞춰보자.


# ============================================


loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# =========================================


cost_history = np.empty(shape=[1], dtype=float)
with tf.Session() as session:
    tf.initialize_all_variables().run()

    for itr in range(total_iterations):
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = tr_features[offset:(offset + batch_size), :, :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]

        _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        # ValueError: Cannot feed value of shape(18, 2) for Tensor 'Placeholder_1:0', which has shape '(?, 10)' 에러가 뜬다.
        # X = Tensor("Placeholder:0", shape=(?, 60, 41, 2), dtype=float32)
        # Y = Tensor("Placeholder_1:0", shape=(?, 10), dtype=float32)
        # 이므로 Y 텐서 차원을 (18,2) 로 바꾸면 되지 않을까? 한번 해 보자.
        cost_history = np.append(cost_history, c)

    # print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
    # print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels})))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history)
    plt.axis([0, total_iterations, 0, np.max(cost_history)])
    plt.show()


# =====================================

# 코드보니까 tr가 입력해서 계산하는 거고, ts와 얼마나 맞는지 맞히는거 인듯..
# 그럼 일단 여자들 yes를 폴더1에 0,1, 폴더 2에 2,3, 폴더 3에 4를 넣어
# 0,1 ,2,3 을 학습, 4와 비교하도록 해보자..
