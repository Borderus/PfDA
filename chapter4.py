import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------------------------------------

# Section 1: Array creation

array11 = np.array([0.9526, -0.246, -0.8856])
array12 = 10*array11

print("Array 11:\n", array11)
print("Array 12:\n", array12)
# We can print the dimensions
print("Array 12 Dimensions:", array12.ndim)
# We can print the shape
print("Array 12 Shape:", array12.shape)
# We can also print the data type
print("Array 12 Datatype:", array12.dtype)

# We can create an array made of 0s, 1s, identity, we can do it without initialising values or a range
array13 = np.zeros(4)
print("Array 13:\n", array13)
array14 = np.ones(4)
print("Array 14:\n", array14)
array15 = np.identity(4)
print("Array 15:\n", array15)
array16 = np.empty(4)
print("Array 16:\n", array16)
array17 = np.arange(4)
print("Array 17:\n", array17)

# DATA TYPES
# Data types come as type and then byte assignments - e.g. float64
# Types: int, uint, float, complex, (bool, object, string_, snicode_ as also available but don't need numbers)

# We can cast between types using the astype method

array18 = array12.astype(np.uint64)
print("Array 18:\n", array18)
# Converting negative numbers to unsigned ints causes underflow

# ------------------------------------------------------------------------------------------------------------

# Section 2: Operations between arrays and scalars

array21 = np.array([[1., 2., 3.],[4., 5., 6.]])
print("Array 21:\n", array21)

array22 = array21*array21
print("Array 22:\n", array22)

array23 = array22 - array21
print("Array 23:\n", array23)

array24 = 1/array21
print("Array 24:\n", array24)

array25 = array21**0.5
print("Array 25:\n", array25)

# Interactions between arrays of different sizes is called BROADCASTING and will be covered later on

# ------------------------------------------------------------------------------------------------------------

# Section 3: Basic Indexing and Slicing

array31 = np.arange(10)
print("Array 31:\n", array31)
print("Array 31 Item 5:\n", array31[5])
print("Array 31 Items 5-7:\n", array31[5:8])
array31[5:8] = 12
print("Altering Array 31 with direct slicing:\n", array31)

# Note: Slices are a view on the array, not a copy of the portion of the array specified

array32 = array31[5:8]
array32[1] = 12345
print("Altering Array 31 using a slice as a view on the array:\n", array31)

# Higher dimensions
array33 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
array34 = array33[1]
print("Array 34:\n", array34)
array35 = array33[1][1:2]
print("Array 35:\n", array35)
# Copying the slice to be able to edit it. Two sections can also be comma delimited
array36 = array33[1, 1:2].copy()
print("Array 36:\n", array36)

# ------------------------------------------------------------------------------------------------------------

# Section 4: Boolean Indexing

array41 = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
print("Array 41:\n", array41)
array42 = np.array(array41 == "Bob")
print("Array 42:\n", array42)
array43 = array41[array42]
print("Array 43:\n", array43)

# ------------------------------------------------------------------------------------------------------------

# Section 5: Fancy Indexing

array51 = np.empty((8,4))

for i in range(8):
    array51[i] = i

print("Array 51:\n", array51)

# We can select a list of desired rows to pull by giving a list:
array52 = array51[[4,3,0,-2]]
print("Array 52:\n", array52)

# We can effectively specify coordinates in the array like this. The following pulls from (4,0), (3,3), (0,1) and (-2,2)
array53 = array51[[4,3,0,-2], [0, 3, 1, 2]]
print("Array 53:\n", array53)

# To get the overlap of these co-ordinate pairs, simplest to us the np.ix_ function
array54 = array51[np.ix_([4,3,0,-2], [0, 3, 1, 2])]
print("Array 54:\n", array54)

# N.B: Fancy Indexing, unlike slicing, copies the data

# ------------------------------------------------------------------------------------------------------------

# Section 6: Transposing arrays and swapping axes

array61 = np.arange(15).reshape(3,5)
print("Array 61:\n", array61)

array62 = array61.T
print("Array 62:\n", array62)

# We can do matrix operations with the dot function. Let's get the inner matrix product (X^T*X)

array63 = np.dot(array61.T, array61)
print("Array 63:\n", array63)

# We can also use the transpose method - this allows us to specify the order in which dimensions need
# swapping too, which is necessary in higher dimensions. There is also the method swapaxes() for just
# swapping two axes.

array64 = array61.transpose()
print("Array 64:\n", array64)

# ------------------------------------------------------------------------------------------------------------

# Section 7: Universal functions

# A universal function, or ufunc, is a function that performs element-wise operations on data in ndarrays

array71 = np.arange(10)
print("Array 71:\n", array71)
# The sqrt and exp functions are examples of these
array72 = np.sqrt(array71)
print("Array 72:\n", array72)
array73 = np.exp(array71)
print("Array 73:\n", array73)
# Add and maximum are universal functions that work on two arrays
array74 = np.array([5,7,2,9,6,1,1,6,0,9])
print("Array 74:\n", array74)
array75 = np.add(array71, array74)
print("Array 75:\n", array75)
array76 = np.maximum(array71, array74)
print("Array 76:\n", array76)

# UNARY:
# abs, fabs: Absolute value. Fabs is faster for non-complex values
# sqrt: Square root
# square: Square
# exp: e^x
# log, log10, log2: Log in bases e, 10 and 2
# sign: Return 1, 0 or -1 depending on sign
# ceil: Ceiling
# floor: Floor
# rint: Round elements to nearest integer, preserving dtype
# modf: Return two arrays, one with integer part of values and one with non-integer part
# isnan: Return boolean array based on whether a value is Not a Number
# isfinite, isinf: Return boolean array dependent on whether value is finite/infinite
# cos, cosh, sin, sinh, tan, tanh, arccos, arccosh, arcsin, arcsinh, arctan, arctanh: Trig
# logical_not: Compute truth value of not current value

# BINARY:
# add: Add
# subtract: Subtract
# multiply: Multiply
# divide, floor_divide: Divide or divide and then floor result
# power: Power
# maximum, fmax: Maximum. Fmax ignores NaN
# minimum, fmin: Minimum. Fmin ignores NaN
# mod: Element-wise modulo
# copysign: Copy sign of values in second array to first
# greater, greater_equal, less, less_equal, equal, not_equal: Comparison
# logical_and, logical_or, logical_xor: Logical operators

# ------------------------------------------------------------------------------------------------------------

# Section 8: Data processing using arrays

array81 = np.arange(-5, 5, 0.1)
print("Array 81:\n", array81)
array82x, array82y = np.meshgrid(array81, array81)
print("Array 82x:\n", array82x)
print("Array 82y:\n", array82y)
array82z = np.sqrt(array82x**2 + array82y**2)

# plt.imshow(array82z, cmap=plt.cm.gray)
# plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# plt.show()

# ------------------------------------------------------------------------------------------------------------

# Section 9: Expressing conditional logic as array operations

# The where function is a good way for establishing what is effectively a case statement

array91 = np.ones(5)
print("Array 91:\n", array91)
array92 = np.arange(5)
print("Array 92:\n", array92)
array93 = np.array([True, False, True, False, False])
print("Array 93:\n", array93)

# Now we try the where function
array94 = np.where(array93, array91, array92)
print("Array 94:\n", array94)
# Arguments can also be scalars
array95 = np.where(array93, 1., array92)
print("Array 95:\n", array95)

# ------------------------------------------------------------------------------------------------------------

# Section 10: Mathematical and Statistical Methods

# Normal distribution, std dev 5 in x direction and 4 in y
array101 = np.random.randn(5,4)
print("Array 101:\n", array101)

# Sum, mean, max and std can be done either by method or function. For example...
print("The mean can be done like this:\n", array101.mean(), "or like this:", np.mean(array101) )
print("The sum can be done like this:\n", array101.sum(), "or like this:", np.sum(array101) )
print("The max can be done like this:\n", array101.max(), "or like this:", np.max(array101) )
print("The deviation can be done like this:\n", array101.std(), "or like this:", np.std(array101) )

# Mean and sum also can take an optional axis argument which calculates a lower-dimensional array
array102 = array101.sum(axis=0)
print("Array 102:\n", array102)

# Other methods like cumsum and cumprod don't aggregate, instead producing a table of intermediate results
array103 = array102.cumsum()
print("Array 103:\n", array103)

# ------------------------------------------------------------------------------------------------------------

# Section 11: Boolean Arrays

# Throw together a random boolean array
array111 = np.array([True, False, True, True, False])
print("Array 111:\n", array111)

# We can count the number of Trues by summing:
array112 = np.sum(array111)
print("Array 112:\n", array112)

# Any and all can be used as or/and on the array as a whole
array113 = np.any(array111)
print("Array 113:\n", array113)
array114 = np.all(array111)
print("Array 114:\n", array114)

# ------------------------------------------------------------------------------------------------------------

# Section 12: Sorting

array121 = np.floor(10*np.random.randn(10, 4))
print("Array 121:\n", array121)
array122 = np.sort(array121)
print("Array 122:\n", array122)
# We can choose which direction to sort in
array123 = np.sort(array121, 0)
print("Array 123:\n", array123)

# N.B: Difference between the function and the method is the function returns a copy and the method does not

# ------------------------------------------------------------------------------------------------------------

# Section 12: Set Logic

# We have a few np.XX() methods to run through here:

# unique(x): Gets the unique elements in x (deduped), sorts them
# intersect1d(x,y): Gets the common elements in x and y, sorts them
# in1d(x,y): Same check as above, returns boolean array
# union1d(x, y): Get the sorted union of elements
# setdiff1d(x, y): Get elements in x but not y (set difference)
# setxor1d(x, y): Get elements not in X and Y (XOR)

# ------------------------------------------------------------------------------------------------------------

# Section 13: File Input and Output with Arrays

# We can save and load arrays with np.save and np.load. These, by default, save the arrays into an
# uncompressed binary format with extension .npy

array131 = np.arange(10)
np.save('datasets/test_array', array131)

array132 = np.load('datasets/test_array.npy')
print("Array 132:\n", array132)

# We can save and load them as txt files too with np.savetxt and np.loadtxt - delimiters can be specified in these

# ------------------------------------------------------------------------------------------------------------

# Section 14: Linear Algebra

# The numpy.linalg library holds numerous functions for performing linear algebra in Python.
# Annoyingly, these were written in FORTRAN - guess I owe Callan a coke

# diag: Convert square matrix to 1D array containing diagonal elements, or 1D array to square matrix with 0s on
#       non-diags.
# dot: Matrix Multiplication
# trace: Sum of diagonal elements
# det: Determinant
# eig: Eigenvalues and eignevectors of square matrix
# inv: Inverse of a square matrix
# pinv: Moore-Penrose psuedo-inverse inverse of a square matrix
# qr: QR decomposition
# svd: Single value decomposition
# solve: Solve linear equation Ax=b for x, where A is a square matrix
# lstsq: Compute least-squares solution to y=Xb

# ------------------------------------------------------------------------------------------------------------

# Section 15: Random Number Generation

# Get 4x4 array of normally distributed samples:
array151 = np.array(np.random.normal(size=(4,4)))
print("Array 151:\n", array151)
# Get random integers
array152 = np.array(np.random.randint(0,10,6))
print("Array 152:\n", array152)
# Get random numbers from [0,1), via uniform distribution
array153 = np.array(np.random.uniform(size=(4,4)))
print("Array 153:\n", array153)

# seed: Seed RNG
# permutation: Return a random permutation of a sequence, or a permuted range
# shuffle: Randomly permute a sequence in place
# rand: Draw samples from a uniform distribution
# randint: Draw random integers from a given high-low range
# randn: Draw samples from a normal distribution with mean 0 and std 1
# binomial: Draw samples from a binomial distribution
# normal: Draw samples from a normal (Gaussian) distribution
# beta: Draw samples from a beta distribution
# chisquare: Draw samples from a chi-square distribution
# gamma: Draw samples from a gamme distribution
# uniform: Draw samples from a uniform [0,1) distribution

# EXAMPLE: Random Walk

# Conventional Way
# position = 0
# walk = []
# xaxis=[]
# steps = 100
# for i in range(steps):
#     if position < -30:
#         break
#     rng_out = np.random.randint(0,2)
#     step = 1 if rng_out else -1
#     position += step
#     xaxis.append(i)
#     walk.append(position)

# plt.plot(xaxis, walk, color='blue')
# plt.show()

# Our way

# noSteps = 200
# walkBool = np.random.randint(0,2,noSteps)
# walkSteps = np.where(walkBool>0, 1, -1)
# walkFull = walkSteps.cumsum()
# xaxis = np.arange(noSteps)

# # With the walk generated, let's calculate some stats
# walkMin = walkFull.min()
# print("Walk min:", walkMin)
# walkMax = walkFull.max()
# print("Walk max:", walkMax)
# # Calculate the first time the walk reaches the number 10. We can do this with boolean logic and 
# # argmax, which records the value of the index when the maximum is first hit
# walkGE10 = (np.abs(walkFull)>=10).argmax()
# print("Walk first hit 10 at:", walkGE10)

# plt.plot(xaxis, walkFull, color='blue')
# plt.show()

# We can use our way to calculate for 5000 walks at once too

noSteps = 200
noWalks = 5000
walkBool = np.random.randint(0,2,size=(noWalks, noSteps))
walkSteps = np.where(walkBool>0, 1, -1)
walkFull = walkSteps.cumsum(1)

# # With the walk generated, let's calculate some stats
walkMin = walkFull.min()
print("Walk min:", walkMin)
walkMax = walkFull.max()
print("Walk max:", walkMax)
# # Let's play with min crossing time now! First we need to wean out those that don't reach 10
hits10 = (np.abs(walkFull) >= 30).any(1)

crossingtimes = (np.abs(walkFull[hits10]) >= 10).argmax(1)
mean_crossing = crossingtimes.mean()
print(mean_crossing)