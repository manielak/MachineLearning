import numpy as np

scratch = np.zeros(10, dtype=int)
scratch2 = np.zeros(10, dtype=np.bool_)
my_ones = np.ones((3,5), dtype=float)
my_full = np.full((3,5), 3.14)
my_arrange = np.arange(0,30, 3)
my_linespace = np.linspace(0,1, 5)
my_random = np.random.random((5,7))
my_normal_distribution = np.random.normal(0,1,(3,3))
my_randint = np.random.randint(0,10,(3,3))
my_eye = np.eye(3,3)
my_empty = np.empty(4)


# my_random = np.random.random((3,3))


# test = np.array([1,4,2,5,3])
# test2 = np.array([3.14, 4, 2, 3], dtype=np.float32)
# test3 = np.array([range(i, i+3) for i in [2,4,6]])

print(scratch2)



# L = list(range(11))
# print(L)
