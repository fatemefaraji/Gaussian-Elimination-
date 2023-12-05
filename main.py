import numpy as np

def gaussian_elimination(A, b):

    A = A.astype(float)
    b = b.astype(float)

    AugmentedMatrix = np.column_stack((A, b))

    n = len(AugmentedMatrix)

    for i in range(n):
        factor = AugmentedMatrix[i, i]
        AugmentedMatrix[i, :] /= factor

        for j in range(i + 1, n):
            factor = AugmentedMatrix[j, i] / AugmentedMatrix[i, i]
            AugmentedMatrix[j, :] -= factor * AugmentedMatrix[i, :]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = AugmentedMatrix[i, -1] - np.dot(AugmentedMatrix[i, i+1:n], x[i+1:])

    return x
'''
n = int(input("Enter the dimension n: "))
print("Enter the elements of the matrix row by row:")
A = np.zeros((n, n), dtype=float)
for i in range(n):
    row = input().split()
    A[i] = [float(x) for x in row]

print("Enter the elements of the vector b (separated by space):")
b_input = input().split()
b = np.array([float(x) for x in b_input])
'''

A = np.array([[2,6,-2],
              [2,4,2],
              [1,2,3]])

n = len(A)
print('A=',A)

b = np.array([6,8,6])
print('\n','b=',b)


x = gaussian_elimination(A, b)

print("Coefficient Matrix A:")
print(A)
print("\nRight-hand side vector b:")
print(b)
print("\nSolution vector x:")
print(x)
