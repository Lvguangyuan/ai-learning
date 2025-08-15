import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return 2*x, 2*y

# 网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# ── 图 1 ──
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=20)      # 等高线
xs = np.linspace(-1.5, 1.5, 11)
ys = np.linspace(-1.5, 1.5, 11)
XS, YS = np.meshgrid(xs, ys)
Gx, Gy = grad_f(XS, YS)
plt.quiver(XS, YS, Gx, Gy, angles="xy")  # 梯度场
plt.title("f(x,y)=x²+y² 的等高线与梯度场")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
plt.show()

# ── 图 2 ──
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=20)
x0, y0 = 1, 1
gx0, gy0 = grad_f(x0, y0)
plt.quiver([x0], [y0], [gx0], [gy0],
           angles="xy", scale_units="xy", scale=1, width=0.01)
plt.scatter([x0], [y0])
plt.text(x0+0.1, y0, f"∇f(1,1)=({gx0},{gy0})")
plt.title("点 (1,1) 处的梯度向量")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
plt.show()
