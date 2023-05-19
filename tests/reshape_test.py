import numpy as np
import torch

arr1 = torch.as_tensor(np.array(
    [[[125,   1]],

        [[126,   1]],

        [[127,   1]],

        [[128,   1]],

        [[129,   1]],

        [[130,   1]],

        [[131,   1]]]
))

arr2 = torch.as_tensor(np.array(
    [[[125,   1]],

        [[126,   1]],

        [[127,   1]],

        [[128,   1]],

        [[129,   1]],

        [[130,   1]],

        [[131,   1]]]
))

arr = torch.cat([arr1, arr2])

print(arr.shape)

a = np.array([1,2,3,4,5,6])

print(a.reshape(2,3))
print(a.reshape(3,2).T)
