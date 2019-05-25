import numpy as np

# ones = np.ones((4,2))
ones = np.array([[1,2],[3,4],[5,6],[7,8]])
print (ones)
new_ones = ones[[1,2],[1,0]]
print (new_ones)
# ones[range(3),1] = 1
ones[[0,1,2,3],1] = 1
print (ones)


#
#
# labels = [9,9,10,10]
#
# n_labels = len(labels)
# n_unique_labels = len(np.unique(labels)) # n_unique_labels = 2
# one_hot_encode = np.zeros((n_labels ,n_unique_labels))
# what_is_arange = np.arange(n_labels)
# print (one_hot_encode)
# print(what_is_arange)
# one_hot_encode[np.arange(n_labels),[1,1,1,1]]
# print (one_hot_encode,labels)
#
# # one_hot_encode[np.arange(n_labels), labels] = 1
