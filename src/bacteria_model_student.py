import numpy as np
import matplotlib.pyplot as plt

class BacteriaModel:
    """
    细菌模型类，用于实现 V(t) 和 W(t) 模型。
    """
    def __init__(self, A, tau):
        """
        初始化模型参数。
        :param A: 模型 W(t) 的幅度参数
        :param tau: 时间常数
        """
        self.A = A
        self.tau = tau

    def v_model(self, t):
        """
        计算 V(t) 模型的值。
        :param t: 时间
        :return: V(t) 的值
        """
        return 1 - np.exp(-t / self.tau)

    def w_model(self, t):
        """
        计算 W(t) 模型的值。
        :param t: 时间
        :return: W(t) 的值
        """
        return self.A * (np.exp(-t / self.tau) - 1 + t / self.tau)


def load_bacteria_data(filepath):
    """
    加载实验数据。
    :param filepath: 数据文件路径
    :return: 时间数据和响应数据
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data[:, 0], data[:, 1]
    except:
        return np.loadtxt(filepath, delimiter=',', unpack=True)


def plot_models_and_data(models, t, time_data=None, response_data=None, title=None):
    """
    绘制模型曲线和实验数据。
    :param models: 模型实例列表
    :param t: 时间序列
    :param time_data: 实验时间数据
    :param response_data: 实验响应数据
    :param title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot(t, model.w_model(t), label=f'A={model.A}, τ={model.tau}')
    
    if time_data is not None and response_data is not None:
        plt.scatter(time_data, response_data, label='Experimental Data', color='black', marker='o')
    
    plt.xlabel('Time (t)')
    plt.ylabel('Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    主函数，整合所有任务。
    """
    # 任务 1.1a: 绘制 W(t) 曲线 (A=1, τ=1)
    model1 = BacteriaModel(A=1.0, tau=1.0)
    t = np.linspace(0, 2, 100)
    plot_models_and_data([model1], t, title='W(t) for A=1.0, τ=1.0')

    # 任务 1.1b: 绘制不同参数的 W(t) 曲线
    model2 = BacteriaModel(A=2.0, tau=1.0)
    model3 = BacteriaModel(A=1.0, tau=2.0)
    plot_models_and_data([model1, model2, model3], t, title='W(t) for Different Parameters')

    # 任务 1.2a: 加载实验数据并绘制 V(t) 和实验数据
    time_data_a, response_data_a = load_bacteria_data('data/g149novickA.txt')
    t = np.linspace(0, 10, 100)
    model_fit_a = BacteriaModel(A=1.5, tau=1.8)
    plot_models_and_data([model_fit_a], t, time_data_a, response_data_a, title='Model Fitting to Experimental Data (g149novickA)')

    # 任务 1.2b: 加载 g149novickB 数据并拟合 W(t)
    time_data_b, response_data_b = load_bacteria_data('data/g149novickB.csv')
    mask = time_data_b <= 10  # 仅保留时间 ≤ 10 小时的数据
    time_data_b = time_data_b[mask]
    response_data_b = response_data_b[mask]
    model_fit_b = BacteriaModel(A=1.2, tau=1.5)
    plot_models_and_data([model_fit_b], t, time_data_b, response_data_b, title='Model Fitting to Experimental Data (g149novickB, t ≤ 10)')


if __name__ == "__main__":
    main()
