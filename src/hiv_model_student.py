import numpy as np
import matplotlib.pyplot as plt

class HIVModel:
    def __init__(self, A, alpha, B, beta):
        """
        初始化模型参数
        
        参数:
            A: 模型参数 A
            alpha: 模型参数 α
            B: 模型参数 B
            beta: 模型参数 β
        """
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def viral_load(self, time):
        """
        计算病毒载量
        
        参数:
            time: 时间数组
            
        返回:
            病毒载量数组
        """
        return self.A * np.exp(-self.alpha * time) + self.B * np.exp(-self.beta * time)

    def plot_model(self, time, label=None):
        """
        绘制模型曲线
        
        参数:
            time: 时间数组
            label: 曲线标签
        """
        viral_load = self.viral_load(time)
        plt.plot(time, viral_load, label=label)

def load_hiv_data(filepath):
    """
    加载HIV数据
    
    参数:
        filepath: 数据文件路径
        
    返回:
        time_data: 时间数据数组
        viral_load_data: 病毒载量数据数组
    """
    try:
        # 尝试加载 .npz 文件
        hiv_data = np.load(filepath)
        time_data = hiv_data['time_in_days']
        viral_load_data = hiv_data['viral_load']
    except:
        # 如果 .npz 文件不存在，尝试加载 .csv 文件
        hiv_data = np.loadtxt(filepath, delimiter=',')
        time_data = hiv_data[:, 0]
        viral_load_data = hiv_data[:, 1]
    return time_data, viral_load_data

def main():
    """
    主函数，用于测试模型
    """
    # 生成时间序列
    time = np.linspace(0, 10, 100)

    # 定义不同的模型参数
    models = [
        HIVModel(A=1, alpha=1, B=0, beta=0),  # 只有 A 和 alpha 起作用
        HIVModel(A=1, alpha=2, B=0, beta=0),  # 增加 alpha
        HIVModel(A=1, alpha=1, B=0.5, beta=0.5),  # 加入 B 和 beta
        HIVModel(A=1, alpha=1, B=0.5, beta=2),  # 增加 beta
    ]

    # 绘制不同参数下的模型曲线
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        model.plot_model(time, label=f"Model {i+1}")
    plt.xlabel('时间 (t)')
    plt.ylabel('病毒载量 (V(t))')
    plt.title('HIV 病毒载量模型')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 加载实验数据
    time_data, viral_load_data = load_hiv_data('HIVseries.npz')  # 或 'HIVseries.csv'

    # 绘制实验数据
    plt.figure(figsize=(10, 6))
    plt.scatter(time_data, viral_load_data, color='blue', label='实验数据', marker='o')
    plt.xlabel('时间 (天)')
    plt.ylabel('病毒载量 (V(t))')
    plt.title('HIV 病毒载量实验数据')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 拟合模型到实验数据
    from scipy.optimize import minimize

    def objective_function(params, time, viral_load):
        A, alpha, B, beta = params
        model = HIVModel(A, alpha, B, beta)
        return np.sum((model.viral_load(time) - viral_load) ** 2)

    # 初始猜测值
    initial_guess = [1, 1, 0.5, 0.5]

    # 拟合模型
    result = minimize(objective_function, initial_guess, args=(time_data, viral_load_data))
    A_fit, alpha_fit, B_fit, beta_fit = result.x

    # 打印拟合结果
    print(f"拟合参数: A={A_fit:.4f}, α={alpha_fit:.4f}, B={B_fit:.4f}, β={beta_fit:.4f}")

    # 绘制拟合结果
    fitted_model = HIVModel(A_fit, alpha_fit, B_fit, beta_fit)
    plt.figure(figsize=(10, 6))
    plt.scatter(time_data, viral_load_data, color='blue', label='实验数据', marker='o')
    fitted_model.plot_model(time_data, label='拟合模型')
    plt.xlabel('时间 (天)')
    plt.ylabel('病毒载量 (V(t))')
    plt.title('HIV 病毒载量模型拟合')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
