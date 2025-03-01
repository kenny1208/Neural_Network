import tensorflow as tf

# 定義 TensorFlow 變數
x = tf.Variable(3.0)
y = tf.Variable(4.0)

# 定義函數 f(x, y)
def f(x, y):
    return x**2 + 2*y

# 使用 tf.GradientTape 計算梯度
with tf.GradientTape(persistent=True) as tape:
    z = f(x, y)

# 取得對 x 和 y 的梯度
grad_x = tape.gradient(z, x)
grad_y = tape.gradient(z, y)

# 釋放記憶體
del tape

# 輸出梯度值
print("梯度（對 x 的偏導數）：", grad_x.numpy())
print("梯度（對 y 的偏導數）：", grad_y.numpy())
