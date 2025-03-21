"""
最小二乘拟合和光电效应实验
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载数据文件
    
    参数:
        filename: 数据文件路径
        
    返回:
        x: 频率数据数组
        y: 电压数据数组
    """
    data = np.loadtxt(filename)
    x = data[:, 0]  # 第一列：频率 (Hz)
    y = data[:, 1]  # 第二列：电压 (V)
    return x, y

def calculate_parameters(x, y):
    """
    计算最小二乘法拟合的斜率和截距
    
    参数:
        x: 自变量数据
        y: 因变量数据
        
    返回:
        m: 斜率
        c: 截距
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("输入数据不能为空")
    
    N = len(x)
    Ex = np.mean(x)
    Ey = np.mean(y)
    Exx = np.mean(x**2)
    Exy = np.mean(x * y)
    
    # 计算斜率和截距
    m = (Exy - Ex * Ey) / (Exx - Ex**2)
    c = (Exx * Ey - Ex * Exy) / (Exx - Ex**2)
    
    return m, c

def calculate_planck_constant(m):
    """
    计算普朗克常量
    
    参数:
        m: 斜率
        
    返回:
        h: 计算得到的普朗克常量值
        relative_error: 与实际值的相对误差(%)
    """
    if not isinstance(m, (int, float)) or np.isnan(m):
        raise ValueError("斜率必须为有效的数值")
    
    e = 1.602e-19  # 电子电荷 (C)
    h_actual = 6.626e-34  # 实际普朗克常量 (J·s)
    
    # 计算普朗克常量
    h = m * e
    
    # 计算相对误差
    relative_error = abs(h - h_actual) / h_actual * 100
    
    return h, relative_error

def plot_data_and_fit(x, y, m, c):
    """
    绘制数据点和拟合直线
    
    参数:
        x: x坐标数组
        y: y坐标数组
        m: 斜率
        c: 截距
    
    返回:
        fig: matplotlib图像对象
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("输入数据不能为空")
    if not isinstance(m, (int, float)) or not isinstance(c, (int, float)):
        raise ValueError("斜率和截距必须为有效的数值")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='blue', label='实验数据')  # 绘制数据点
    ax.plot(x, m * x + c, color='red', label='拟合直线')  # 绘制拟合直线
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('电压 (V)')
    ax.set_title('光电效应实验数据与拟合直线')
    ax.legend()
    ax.grid(True)
    return fig

def main():
    """主函数"""
    # 数据文件路径
    filename = "millikan.txt"
    
    # 加载数据
    x, y = load_data(filename)
    
    # 计算拟合参数
    m, c, Ex, Ey, Exx, Exy = calculate_parameters(x, y)
    
    # 打印结果
    print(f"Ex = {Ex:.6e}")
    print(f"Ey = {Ey:.6e}")
    print(f"Exx = {Exx:.6e}")
    print(f"Exy = {Exy:.6e}")
    print(f"斜率 m = {m:.6e}")
    print(f"截距 c = {c:.6e}")
    
    # 绘制数据和拟合直线
    fig = plot_data_and_fit(x, y, m, c)
    
    # 计算普朗克常量
    h, relative_error = calculate_planck_constant(m)
    print(f"计算得到的普朗克常量 h = {h:.6e} J·s")
    print(f"与实际值的相对误差: {relative_error:.2f}%")
    
    # 保存图像
    fig.savefig("millikan_fit.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
