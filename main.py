import numpy as np


def toFixedMax(y_max, digits=0):
    return f"{y_max:.{digits}f}"


def toFixedMin(y_min, digits=0):
    return f"{y_min:.{digits}f}"


def toFixed(my, digits=0):
    return f"{my :.{digits}f}"


def out_yellow(text):
    print("\033[33m {}".format(text))


N = 4
m = 3
q = 0.05

variant = np.array([[-15, 30], [-35, 15], [-25, 5]])

x_cp_min, x_cp_max = variant.T[0].mean(), variant.T[1].mean()

intY = True

value_a = 0.7679
value_b = 2.306
value_c = 4.5

massive_p = []

y_max = 200 + x_cp_max
y_min = 200 + x_cp_min

factors = []
for i in range(2 ** variant.shape[0]):
    factors.append([int(j) for j in '{0:0{fmt}b}'.format(i, fmt=variant.shape[0])])

# rng = range(2 ** variant.shape[0])

in_matrix = np.random.choice(2 ** variant.shape[0], 2 ** variant.shape[0] - N, replace=False)
pop_elem = [factors[i] for i in in_matrix]
for i in pop_elem:
    factors.remove(i)
aft = np.copy(factors)
for i in factors:
    i.insert(0, 1)
factors = np.array(factors)
factors[factors == 0] = -1

aft = aft.T
for i in range(variant.shape[0]):
    arr = aft[i]
    for j in range(len(arr)):
        if arr[j] == 1:
            arr[j] = variant[i][1]
        else:
            arr[j] = variant[i][0]
    aft[i] = arr
aft = aft.T

if intY:
    y = np.random.randint(y_min, y_max + 1, (N, m))
else:
    y = np.random.sample((N, m)) * (y_max - y_min) + y_min

y_mean = [
    i.mean() for i in y
]

x_mean = [
    i.mean() for i in aft.T
]

my = np.mean(y_mean)

a1, a2, a3 = 0, 0, 0
a11, a22, a33 = 0, 0, 0
a12, a13, a23 = 0, 0, 0
j = 0

for i in aft:
    a1 += i[0] * y_mean[j]
    a2 += i[1] * y_mean[j]
    a3 += i[2] * y_mean[j]
    a11 += i[0] ** 2
    a22 += i[1] ** 2
    a33 += i[2] ** 2
    a12 += i[0] * i[1] / N
    a13 += i[0] * i[2] / N
    a23 += i[1] * i[2] / N
    j += 1
a1, a2, a3 = a1 / N, a2 / N, a3 / N
a11, a22, a33 = a11 / N, a22 / N, a33 / N
a21 = a12
a31 = a13
a32 = a23

initialMatrix = [
    [1, x_mean[0], x_mean[1], x_mean[2]],
    [x_mean[0], a11, a12, a13],
    [x_mean[1], a12, a22, a32],
    [x_mean[2], a13, a23, a33]
]

det_m = np.linalg.det(initialMatrix)

y_through_std = [np.std(i) for i in y]

g_massive_p = np.max(y_through_std) / np.sum(y_through_std)
f1 = m - 1
f2 = N

# f3 and t

y_std_mean = np.mean(y_through_std)
S_2b = y_std_mean / (N * m)
t = []
for i in range(N):
    t.append(np.abs(np.sum(y_mean * factors.T[i]) / N) / S_2b)

print(f"y_max = {toFixedMax(y_max, 3)}")

print(f"y_min = {toFixedMin(y_min, 3)}")

print("-" * 20 + "factors" + "-" * 20)
print(factors)

print("-" * 20 + "ft  -> after_factors" + "-" * 20)
print(aft)

print("-" * 20 + "y" + "-" * 20)
print(y)

print("-" * 20 + "y_mean" + "-" * 20)
for i in y_mean:
    print('%.3f' % i)

print("-" * 20 + "x_mean" + "-" * 20)
for i in x_mean:
    print('%.3f' % i)

print("-" * 20 + "my" + "-" * 20)
print(f"{toFixed(my, 3)}")

print("-" * 20 + "a with coefficients" + "-" * 20)
print(f"{a1} \n {a2} \n {a3} \n {a11} \n {a22} \n {a33} \n {a12} \n {a13} \n {a31}")

print("-" * 20 + "initial matrix" + "-" * 20)
print(
    f" \t {[1, x_mean[0], x_mean[1], x_mean[2]]} \n "
    f"{[x_mean[0], a11, a12, a13]} \n "
    f"{[x_mean[1], a12, a22, a32]} \n "
    f"{[x_mean[2], a13, a23, a33]} \n ")

print("-" * 20 + "det_m of initial" + "-" * 20)
print(toFixed(det_m, 3))

print("-" * 20 + "clear matrix" + "-" * 20)
to_replace = [my, a1, a2, a3]
for i in range(N):
    clear_matrix = np.copy(initialMatrix)
    clear_matrix = np.array(clear_matrix)
    print(f"{clear_matrix} \n")
    # can be used ( replace ), but the numbers will be large even after 3 digits
    massive_p.append(np.linalg.det(clear_matrix) / det_m)

print("-" * 20 + "Рівняння регресії" + "-" * 20)
print(
    "y = "
    f"{round(massive_p[0], 2)}"
    f" + {round(massive_p[1], 2)} * x1 + "
    f"{round(massive_p[2], 2)} * x2 + "
    f"{round(massive_p[3], 2)} * x3"
)

comparison = [massive_p[0] + massive_p[1] * i[0] + massive_p[2] * i[1] + massive_p[3] * i[2] for i in aft]
print("-" * 25 + "Compare" + "-" * 25)
print(np.vstack((comparison, y_mean)))

print("-" * 20 + "factors by Y" + "-" * 20)
print((np.hstack((factors, y))))

print("-" * 25 + "y_std" + "-" * 25)
for i in y_through_std:
    print('%.3f' % i)

print("-" * 25 + "values" + "-" * 25)
print(
    f"g_m_p = {round(g_massive_p, 5)} \n "
    f"f1 = {f1} \n"
    f"f2 = {f2}"
)

print("-" * 15 + "determining the variance" + "-" * 15)
if g_massive_p < value_a:
    print("Дисперсія однорідна")
elif g_massive_p > value_a:
    m += 1
    print(f" додаємо 1 до m щоб дисперсія стада однорідною. \n m + 1 = {m}, "
          "Дисперсія однорідна")
else:
    print("Дисперсія неоднорідна")

print("-" * 25 + "f3" + "-" * 25)
print(f"f3 = {f1 * f2}")

dev = ""
to_save = []
for i in range(N):
    if t[i] > value_b:
        dev = dev + f"b{i} * x{i}"
        added = True
        to_save.append(1)
    else:
        print(f"Коефіцієнт b{i} - незначимий")
        added = False
        to_save.append(0)
    if i != 3 and added:
        dev = dev + " + "
print("-" * 10 + "t, Рівняння регресії, comparison by y_mean" + "-" * 10)
print("-" * 5 + "t" + "-" * 5)
for i in t:
    print('%.3f' % i)

print("-" * 5 + "рівняння" + "-" * 5)
print(f"y = {dev}")

print("-" * 5 + "comparison" + "-" * 5)
d = to_save.count(1)

for i in range(N):
    if to_save[i] == 0:
        massive_p[i] = 0
comparison = [massive_p[0] + massive_p[1] * i[0] + massive_p[2] * i[1] + massive_p[3] * i[2] for i in aft]
print(np.vstack((comparison, y_mean)))

out_yellow("-" * 25 + "Перевірка" + "-" * 25)
print(f"d = {d}")
if d == N:
    print("Отже рівняння регресії адекватно за критерієм ")
else:
    f4 = N - d
    S_2_ad = m / (N - d) * np.sum((np.array(y_mean) - np.array(comparison)) ** 2)
    value_m = S_2_ad / S_2b
    value_c = [5.3, 4.5, 4.1, 3.8]
    if value_m < float(value_c[f4 - 1]):
        print("Рівняння регресії адекватно за критерієм")
    else:
        print("Рівняння регресії неадекватно за критерієм")
