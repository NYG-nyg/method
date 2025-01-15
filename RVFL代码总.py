import math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys

# 从Excel表中读取数据
W = np.array([1/7,1/7,1/7,1/7,1/7,1/7,1/7])
len_e = 90   #决策者数量
class1 = {} #存放聚类结果
Y =[]       #存放每组评价矩阵
      #聚类数量
reach_consensus = 0.9
lambda_val = 0.5
Max=10
lamda_val2=2.25
alpha=1.21
beta=1.02
data = pd.read_excel('1.xlsx')
# 定义模糊化处理函数
import pandas as pd
import numpy as np

# 从Excel表中读取数据
data = pd.read_excel('1.xlsx')
num_complete_matrices = data.shape[0] // 7
complete_matrices = data.values[:num_complete_matrices * 7].reshape(num_complete_matrices, 7, 5)
transposed_complete_matrices = complete_matrices.transpose(0, 2, 1)
transposed_data = pd.DataFrame(transposed_complete_matrices.reshape(-1, transposed_complete_matrices.shape[2]))
header = ["列1", "列2", "列3", "列4", "列5", "列6", "列7"]
transposed_data_with_header = pd.DataFrame(transposed_complete_matrices.reshape(-1, transposed_complete_matrices.shape[2]), columns=header)
output_file_path_complete = 'transposed.xlsx'
transposed_data_with_header.to_excel(output_file_path_complete, index=False)
data=pd.read_excel('transposed.xlsx')
# 定义模糊化处理函数
def fuzzy_process(data):
    data_np = data.to_numpy()  # 将Pandas DataFrame转换为NumPy数组
    data_min = np.min(data_np)
    data_max = np.max(data_np)
    if data_max == data_min:
        return np.zeros_like(data_np)  # 返回全零数组或其他适当响应
    return (data_np - data_min) / (data_max - data_min)
# 计算隶属度函数
def membership_function(mu_ij, lambda_val):
    return 1 - (1 - mu_ij) / (1 - (1 - np.exp(lambda_val)) * mu_ij)
# 计算非隶属度函数
def non_membership_function(mu_ij, lambda_val):
    return (1 - mu_ij) / (1 - (1 - np.exp(lambda_val + 1)) * mu_ij)
# 计算犹豫度函数
def hesitation_degree(mu_ij, v_ij):
    return pow(1 - mu_ij ** 5 - v_ij ** 5,1/5)
# 定义 lambda 值
lambda_val = 0.5

# 生成n个矩阵，每个矩阵存储隶属度和非隶属度
n = data.shape[0] // 5  # 计算问卷数量
matrices = []  # 存储生成的矩阵
for i in range(n):
    start_idx = i * 5
    end_idx = start_idx + 5
    sub_data = data.iloc[start_idx:end_idx]  # 提取每份问卷数据
    mu_ij = fuzzy_process(sub_data)  # 模糊化处理
    mu_ij_qfs= membership_function(mu_ij, lambda_val)  # 计算隶属度
    v_ij_qfs = non_membership_function(mu_ij_qfs, lambda_val)  # 计算非隶属度

    matrix = np.zeros((5, 7, 2))  # 创建一个7x5的矩阵，每个元素存储两个值
    for row in range(5):
        for col in range(7):
            matrix[row, col, 0] = mu_ij_qfs[row, col]  # 存储隶属度
            matrix[row, col, 1] = v_ij_qfs[row, col]  # 存储非隶属度
    matrices.append(matrix)
# 定义得分函数和精度函数
def score_function(mu, v, q):
    return mu**q - v**q
def accuracy_function(mu, v, q):
    return mu**q + v**q

q = 5
xinmatrices = []
for matrix_num, matrix_data in enumerate(matrices):
    dim_reduced_data = np.zeros((5, 7))
    for i in range(5):
        for j in range(7):
            mu = matrix_data[i, j, 0]
            v = matrix_data[i, j, 1]
            dim_reduced_data[i, j] = score_function(mu, v, q) + accuracy_function(mu, v, q)
    xinmatrices.append(dim_reduced_data)
def calculate_euclidean_distance(x, centroid):
    return np.linalg.norm(x - centroid)

def calculate_hausdorff_distance(data_point, centroid, data):
    dists = np.array([np.linalg.norm(data_point - other) for other in data])
    return np.max(dists)

def combined_distance(data_point, centroid, data):
    return calculate_euclidean_distance(data_point, centroid) + calculate_hausdorff_distance(data_point, centroid, data)


def assign_clusters_with_threshold(data, centroids, threshold=9.3):
    clusters = {i: {'core_indices': [], 'boundary_indices': []} for i in range(len(centroids))}

    for i, x in enumerate(data):
        distances = [combined_distance(x, centroid, data) for centroid in centroids]
        min_distance_index = np.argmin(distances)

        # 根据最小距离判断数据点是否在核心域
        if distances[min_distance_index] < threshold:
            clusters[min_distance_index]['core_indices'].append(i)
        else:
            # 将点加入边界域
            clusters[min_distance_index]['boundary_indices'].append(i)

    return clusters
def kmeans_plus_plus_init(X, k):
    n_samples, n_features = X.shape
    centroids = [X[np.random.randint(n_samples)]]
    D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])

    for _ in range(1, k):
        probs = D2 / D2.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(X[i])
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])

    return np.array(centroids)

def kmeans_plus_plus(X, k):
    # 初始化质心
    centroids = kmeans_plus_plus_init(X, k)

    # 初始化簇分配
    assignments = np.zeros(X.shape[0])

    while True:
        # 保存上一次迭代的质心
        old_centroids = centroids.copy()

        # 计算每个点到质心的距离，并更新簇分配
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)

        # 计算新的质心
        for i in range(k):
            if np.any(assignments == i):
                centroids[i] = np.mean(X[assignments == i], axis=0)
            else:
                centroids[i] = old_centroids[i]  # 如果没有分配给该质心的点，保持原质心

        # 检查收敛条件，即新旧质心是否一致
        if np.all(centroids == old_centroids):
            break

    return assignments, centroids


# 假设xinmatrices是你的输入数据，形状为(n_samples, n_features)
xinmatrices_array = np.array(xinmatrices)  # 请确保xinmatrices已初始化
xinmatrices_array_flattened = xinmatrices_array.reshape(xinmatrices_array.shape[0], -1)

# 选择聚类中心的数量
k = 5
# 使用k-means++初始化聚类中心
centroids = kmeans_plus_plus_init(xinmatrices_array_flattened, k)
# 分配数据点到聚类
clusters_three_way = assign_clusters_with_threshold(xinmatrices_array_flattened, centroids)
classt={}
# 输出聚类结果索引
for cluster_id, cluster_info in clusters_three_way.items():
    core_indices = cluster_info.get('core_indices', [])
    boundary_indices = cluster_info.get('boundary_indices', [])
    core_indices0=[i+1 for i in core_indices]
    boundary_indices0=[i+1 for i in boundary_indices]
    # classt[cluster_id+1]={}
    # classt[cluster_id+1]['core_indices']=core_indices0
    # classt[cluster_id+1]['boundary_indices']=boundary_indices0
class1={1:{'core_indices': [79], 'boundary_indices': [2, 3, 4, 5, 19, 20, 23, 35, 36, 38, 49, 56, 62, 66, 70, 71, 72, 76, 78, 81, 87, 88, 89]},2:{'core_indices': [44, 46], 'boundary_indices': [8, 9, 10, 11, 12, 13, 14, 15, 17, 24, 27, 28, 29, 30, 31, 37, 39, 40, 41, 42, 43, 45, 47, 53, 55, 57, 58, 59, 60, 67, 80, 83]},3:{'core_indices':  [63], 'boundary_indices': [6, 7, 16, 21, 22, 25, 50, 54, 64, 69, 73, 74, 77, 82, 84, 85, 86]},4:{'core_indices': [61], 'boundary_indices':  [65, 75]},5:{'core_indices':  [1], 'boundary_indices':[18, 26, 32, 33, 34, 48, 51, 52, 68, 90]}}
# class1=classt
e=np.copy(xinmatrices)

def calculate_angle1(DM1, DM2):
    # 使用余弦定理计算角度
    c=np.array(DM1)-np.array(DM2)
    a = vector_magnitude(DM1)  # ||DM1||
    b = vector_magnitude(DM2)  # ||DM2||
    m = vector_magnitude(c)
    A1=math.degrees(math.acos((b**2 + m**2 - a**2) / (2 * b * m)))
    A2=math.degrees(math.acos((a**2 + m**2 - b**2) / (2 * a * m)))
    A3=math.degrees(math.acos((a**2 + b**2 - m**2) / (2 * a * b)))
    d=A3/A1-A2/A1
    if d<=-2:
      return 0
    if d>-2 and d<2:
      return 3.14159265358979323846 * (1 + d / 2) / 2
    if d>2:
      return 3.14159265358979323846
def vector_magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

class RVFL(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=100, alpha=1.0, input_scaling=1.0):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.input_scaling = input_scaling

    def fit(self, X, y):
        # 输入数据缩放
        X_scaled = X * self.input_scaling
        self.W_in = np.random.randn(X_scaled.shape[1], self.n_hidden)  # 随机初始化输入到隐藏层权重
        self.B = np.random.randn(self.n_hidden)  # 随机初始化隐藏层偏置
        hidden_output = self._activation(X_scaled @ self.W_in + self.B)  # 隐藏层输出
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(hidden_output, y)

    def predict(self, X):
        X_scaled = X * self.input_scaling
        hidden_output = self._activation(X_scaled @ self.W_in + self.B)
        return self.ridge.predict(hidden_output)

    def _activation(self, x):
        return np.maximum(0, x)  # ReLU激活函数

    def get_weights(self):
        return self.ridge.coef_  # 输出层权重

def chimp_optimization(X_train, y_train, X_val, y_val, num_chimps=20, num_iterations=10,
                       n_hidden_bounds=(50, 500), alpha_bounds=(0.01, 10), scaling_bounds=(0.1, 5.0)):
    # 初始化种群位置 (n_hidden, alpha, scaling)
    population = np.random.rand(num_chimps, 3)
    population[:, 0] = population[:, 0] * (n_hidden_bounds[1] - n_hidden_bounds[0]) + n_hidden_bounds[0]
    population[:, 1] = population[:, 1] * (alpha_bounds[1] - alpha_bounds[0]) + alpha_bounds[0]
    population[:, 2] = population[:, 2] * (scaling_bounds[1] - scaling_bounds[0]) + scaling_bounds[0]

    fitness = np.inf * np.ones(num_chimps)
    best_solution = None
    best_fitness = np.inf

    # 算法参数
    a = 2  # 控制参数，随迭代逐渐减少

    for iteration in range(num_iterations):
        for i in range(num_chimps):
            n_hidden = int(population[i, 0])
            alpha = population[i, 1]
            scaling = population[i, 2]

            model = RVFL(n_hidden=n_hidden, alpha=alpha, input_scaling=scaling)
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            val_loss = np.mean((val_predictions - y_val) ** 2)
            fitness[i] = val_loss

            # 更新最优解
            if val_loss < best_fitness:
                best_fitness = val_loss
                best_solution = population[i].copy()

        # 更新位置
        for i in range(num_chimps):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a  # 攻击参数
            C = 2 * r2  # 距离控制参数

            if i != np.argmin(fitness):  # 不更新最优个体
                leader = best_solution
                distance = np.abs(C * leader - population[i])
                new_position = leader - A * distance
                population[i, :] = new_position

                # 边界处理
                population[i, 0] = np.clip(population[i, 0], n_hidden_bounds[0], n_hidden_bounds[1])
                population[i, 1] = np.clip(population[i, 1], alpha_bounds[0], alpha_bounds[1])
                population[i, 2] = np.clip(population[i, 2], scaling_bounds[0], scaling_bounds[1])

        # 缩小 a
        a -= 2 / num_iterations

    return best_solution, best_fitness

# 定义权重计算函数
def calculate_decision_maker_weights(model, inputs, targets):
    predictions = model.predict(inputs)
    errors = np.abs(predictions - targets)
    contribution = 1 / (errors + 1e-6)
    normalized_weights = contribution / np.sum(contribution)
    return normalized_weights
def consensus_level(X,Y):
    temp = 1-np.abs(X-Y)
    temp = temp*W
    return np.sum(temp)/5
def non_membership_function(mu_ij, lambda_val):
    return (1 - mu_ij) / (1 - (1 - np.exp(lambda_val + 1)) * mu_ij)
# 计算犹豫度函数
def hesitation_degree(mu_ij, v_ij):
    return pow(1 - mu_ij ** 5 - v_ij ** 5,1/5)
def hesitangcy(matrix,q):
    matrxi1=[]
    sum=0
    for i  in matrix:
        u=0
        v=0
        matrix2=[]
        for j in i:
            u=(j/2)**(1/q)
            v=non_membership_function(u, lambda_val)
            j=hesitation_degree(u,v)
            matrix2.append(j)
            sum=sum+j
        matrxi1.append(matrix2)
    return sum
def evaluation_update(X,Y,position):
    x = X[position[0]][position[1]]
    y = Y[position[0]][position[1]]
    risk_aversion = 0.88

    u = abs(x**risk_aversion-y**risk_aversion)
    regret_aversion = 0.6
    R = 1 - np.exp(regret_aversion * u)
    CM1 = consensus_level(X,Y)
    parameter = 50
    modification = 1.0/(1-parameter*(1-CM1)*R)
    if x-y>0:
        x = modification*x - (1-modification)*y
    if x-y<0:
        x = modification*x + (1-modification)*y
    X[position[0]][position[1]]=x
    return X

def hausdorff_distance(X,Y):
    distance_X_to_Y = np.max(np.min(np.abs(X[:, np.newaxis] - Y), axis=1))
    distance_Y_to_X = np.max(np.min(np.abs(Y[:, np.newaxis] - X), axis=1))
    return max(distance_X_to_Y, distance_Y_to_X)
def mixed_distance(X, Y):
    euclidean_dist = np.linalg.norm(X - Y)
    hausdorff_dist = hausdorff_distance(X, Y)
    mixed_dist = 0.5 * euclidean_dist + 0.5 * hausdorff_dist
    return mixed_dist
def DM_Change(X,Y):
    temp = np.abs(X-Y)
    e_old = X
    for i in range(35):
        index = np.argsort(temp.flatten())
        position = np.unravel_index(index[-i],X.shape)
        # print(position)
        X = evaluation_update(X,Y,position)
    return  X
def DM_Change1(X,Y,P):
    temp = np.abs(X-Y)
    e_old = X
    f=np.zeros((5,7))
    g=np.zeros((5,7))
    V=np.zeros((5,7))
    for i in range(5):
        for j in range(7):
            f[i][j]=e_old[i][j]-Y[i][j]
    for i in range(5):
        for j in range(7):
            if f[i][j] >= 0:
                g[i][j] = f[i][j]**alpha
            else:
                g[i][j] = -2.25 * ((-(f[i][j])) ** 1.02)
    w=np.zeros((5,7))
    for i in range(5):
        for j in range(7):
            w[i][j]=P[i]*1/7
    w=w/np.sum(w)
    for i in range(5):
        for j in range(7):
            V[i][j]=g[i][j]*w[i][j]
    max_value = np.max(V)
    min_value = np.min(V)
    R1 = np.zeros((5, 7))
    G1 = np.zeros((5, 7))
    for i in range(5):
        for j in range(7):
            R1[i][j] = 1 - math.exp(0.6* abs((V[i][j] - max_value) / (max_value - min_value)))
    for i in range(5):
        for j in range(7):
            G1[i][j] = 1 - math.exp(-0.6 * abs((V[i][j] - min_value) / (max_value - min_value)))
    Zong=np.zeros((5,7))
    for i in range(5):
        for j in range(7):
            Zong[i][j]=R1[i][j]+G1[i][j]
    elements_with_positions = []
    # 提取每个元素及其位置
    for i in range(5):
        for j in range(7):
            elements_with_positions.append((Zong[i, j], (i, j)))
    # 对元素进行从大到小排序

    elements_with_positions.sort(key=lambda x: x[0])
    # 只保留排序后的原始位置
    position0 = [position for value, position in elements_with_positions]
    # 输出排序后的结果
    for i in range(1,35):
        # index = np.argsort(temp.flatten())
        # position = np.unravel_index(index[-i],X.shape)
        position1=position0[i]
        X = evaluation_update(X,Y,position1)
    return  X
class0={1:{'core_indices': [61], 'boundary_indices': [23,46,60,65,77,78]},2:{'core_indices': [7,8,13,26,27,28,41,45,52], 'boundary_indices': [4,9,10,11,12,14, 15, 16,20,22,24,25,29,30, 35,36,37,38,39,40,42, 53, 54,55,59]},3:{'core_indices': [56,58], 'boundary_indices': [43,44,57]},4:{'core_indices': [5,6,51,63], 'boundary_indices': [0,1,2,17,18,19,21,31,32,33,34,47,48,49,50,62,66,68, 69,72,74 ,76]},5:{'core_indices': [71], 'boundary_indices':[3,64,67,70,73,75] }}
class1={}
for key in class0:
    dict000=class0[key]
    class1[key]=dict000['core_indices']+dict000['boundary_indices']
expert={}
global d
d=0
GA=[]
DM_number=[]
def CRP_algorithm() :
    DM_Weight = {}
    global Y
    global DM_weight
    global expert0
    global matrices3
    DM_weight={}
    Y={}
    for key in class0.keys():
        y=np.zeros((5,7))
        dict_1 = class0[key]
        list1= dict_1['core_indices'] + dict_1['boundary_indices']
        matrices1=[]
        for i in list1:
            matrices1.append(e[i-1])
        Y0=np.zeros((5,7))
        for i in list1:
            for j in range(5):
                for k in range(7):
                    Y0[j][k]+=e[i-1][j][k]
        Y0=Y0/len(list1)
        row_means = np.mean(Y0, axis=1)
        row_means_list = row_means.tolist()
        # 创建训练和测试掩码
        YG0 = []
        for i in range(len(list1)):
            YG0.append(e[list1[i]-1])
        YG0_a=np.array(YG0)
        target_scores = np.random.rand(len(list1))
        flattened_inputs = YG0_a.reshape(len(list1), -1)
        X_train, X_val, y_train, y_val = train_test_split(flattened_inputs, target_scores, test_size=0.2,
                                                          random_state=42)
        # 使用CHOA优化RVFL超参数
        optimal_params, optimal_val_loss = chimp_optimization(X_train, y_train, X_val, y_val)
        # 使用最优参数训练最终模型
        final_model = RVFL(n_hidden=int(optimal_params[0]), alpha=optimal_params[1], input_scaling=optimal_params[2])
        final_model.fit(flattened_inputs, target_scores)
        # 计算决策者权重
        expert0= calculate_decision_maker_weights(final_model, flattened_inputs, target_scores)
        for i in range(len(list1)):
            if list1[i] in dict_1['boundary_indices']:
                expert0[i]=expert0[i]*consensus_level(e[list1[i]-1],Y0)/0.9
        ww=0
        for i in range(len(list1)):
          ww+=expert0[i]
        for i in range(len(list1)):
          expert0[i]=expert0[i]/ww # 计算决策者权重
        for i in  range(len(list1)):
            y+=matrices1[i]*expert0[i]
        Y[key]=y
    print("t=1")
    print(Y)
    #计算决策者e[i]与组Y[key]共识水平
    CMRS = {}
    for key11,content in class1.items():
        for i in content:
            if key11 not in CMRS:
                CMRS[key11] = []
            CMRS[key11].append(consensus_level(e[i-1],Y[key11]))
    #计算每组权重
    Group_weight = []
    sum_weight = 0
    for content in CMRS.values():
        sum_weight += np.sum(content)
    for key12 in CMRS.keys():
        Group_weight.append(np.sum(CMRS[key12])/sum_weight)
    CCAr = {} #每组共识水平
    for key13, content in CMRS.items():
        CCAr[key13] = np.sum(content) / len(content)
    print("初始共识水平")
    print(CCAr)
    dd=0
    for key,content in CCAr.items():
        t=0
        list2 = class1.get(key, [])
        while (CCAr[key]<reach_consensus) and (t<Max):
            if key not in GA:
                GA.append(key)
            dd += 1
            for i in class1[key]:
                t1 = 0
                while consensus_level(e[i-1],Y[key])<reach_consensus  and (t1<Max):
                    if i not in DM_number:
                        DM_number.append(i)
                    e[i-1]=DM_Change(e[i-1],Y[key])
                    t1+=1
                    y2 = np.zeros((5, 7))
                    matrices2 = []
                    for i in list2:
                        matrices2.append(e[i - 1])
                    YG1 = []
                    for i in range(len(list2)):
                        YG1.append(e[list2[i] - 1])
                    YG1_a = np.array(YG1)
                    target_scores = np.random.rand(len(list2))
                    flattened_inputs = YG1_a.reshape(len(list2), -1)
                    X_train, X_val, y_train, y_val = train_test_split(flattened_inputs, target_scores, test_size=0.2,
                                                                      random_state=42)
                    # 使用CHOA优化RVFL超参数
                    optimal_params, optimal_val_loss = chimp_optimization(X_train, y_train, X_val, y_val)
                    # 使用最优参数训练最终模型
                    final_model = RVFL(n_hidden=int(optimal_params[0]), alpha=optimal_params[1],
                                       input_scaling=optimal_params[2])
                    final_model.fit(flattened_inputs, target_scores)
                    # 计算决策者权重
                    expert0 = calculate_decision_maker_weights(final_model, flattened_inputs, target_scores)
                    # print(len(expert0))
                    # print(key)
                    # 计算决策者权重
                    for i00 in range(len(list2)):
                        y2 += matrices2[i00] * (expert0[i00])
                    Y[key] = y2
                # 计算决策者e[i]与组Y[key]共识水平
            CMRS = {}
            for key, content in class1.items():
                for k in content:
                     if key not in CMRS:
                       CMRS[key] = []
                CMRS[key].append(consensus_level(e[k - 1], Y[key]))
            sum_weight=0
            for key, content in CMRS.items():
                CCAr[key] = np.sum(content) / len(content)
            t += 1
    #计算各组权重
    print("第一阶段结束总体共识水平：")
    print(CCAr)
    total=0
    Group_weight={} #五组小组权重
    for key in CMRS.keys():
        total+=np.sum(CMRS[key])
    for key0 in class1.keys():
        Group_weight[key0]=np.sum(CMRS[key0])/total
    print("第一阶段结束各组权重：")
    print(Group_weight)
    arg_min=[]
    for i000 in range(5):
        h=0
        min0=1000000
        h1=0
        for j in range(len(class1[i000+1])):
            h = class1[i000+1][j]
            n=hesitangcy(e[h-1],5)
            if min0 >n:
                min0=n
                h1=h
        arg_min.append(h1)
    print("第二阶段领导者：")
    print(arg_min)
    CMRS=[]
    Y1=np.zeros((5,7))
    c1=1
    for i111 in range(len(arg_min)):
        Y1 += e[arg_min[i111] - 1]*Group_weight[i111+1]
    for i13 in arg_min:
        n=consensus_level(e[i13-1],Y1)
        CMRS.append(n)
    CCAr1=0
    for i14 in range(len(CMRS)):
        CCAr1+=CMRS[i14]*Group_weight[i14+1]
    Y2=np.zeros((5,7))
    for i15 in range(len(arg_min)):
        Y2+=e[arg_min[i15]-1]*Group_weight[i15+1]
    t=0
    ddd=0
    while (CCAr1<reach_consensus) and (t<Max):
            ddd += 1
            for i in arg_min:
                t1=0
                P2=[0,0,0,0,0]
                while consensus_level(e[i-1],Y2)<reach_consensus and (t1<Max):
                        if i not in DM_number:
                          DM_number.append(i)
                        P_ADMK = []
                        for i2 in arg_min:
                            tt = 0
                            # print(e[i - 1][0])
                            row_sums = [sum(row) for row in e[i2 - 1]]
                            row_sums1 = np.sum(e[i2 - 1])
                            tt = row_sums / row_sums1
                            P_ADMK.append(tt)
                        DM = {}
                        for i3 in range(len(arg_min)):
                            DM[i3] = []
                            for j in range(5):
                                s = 0
                                product = Group_weight[i3 + 1] * P_ADMK[i3][j]
                                s = (np.sqrt(product)) ** 2
                                # print(s)
                                DM[i3].append(s)
                        # print(DM)
                        beta0 = []
                        for i4 in range(len(arg_min)):
                            for j in range(i4 + 1, len(arg_min)):
                                beta0.append(calculate_angle1(DM[i4], DM[j]))
                        # print(beta0)
                        P = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                        s0 = 0
                        for i5 in range(5):
                            sum0 = 0
                            for k in range(len(arg_min)):
                                sum0 += math.sqrt(Group_weight[k + 1] * P_ADMK[k][i5])
                            s0 += (sum0 ** 2)
                        alpha1 = 1 / s0
                        for key in P.keys():
                            P1 = 0
                            P22 = 0
                            for i6 in range(len(arg_min)):
                                P1 += DM[i6][key - 1]
                            s = 0
                            for j in range(len(arg_min) - 1):
                                for k in range(j + 1, len(arg_min)):
                                    P22 += math.sqrt(
                                        Group_weight[j + 1] * P_ADMK[j][key - 1] * Group_weight[k + 1] * P_ADMK[k][key - 1]) * \
                                          beta0[s]
                                    s += 1
                            P[key] = alpha1 * (P1 + 2 * P22)
                        # print(P)
                        P1 = [x / sum(P.values()) for x in P.values()]
                        # print(e[i-1])
                        e[i-1]=DM_Change1(e[i-1],Y2,P1)
                        # print(e[i-1])
                        t1+=1
                        matrices3=[]
                        for i1 in arg_min:
                            matrices3.append(e[i1 - 1])
                        # print("2389611")
                        # print(matrices3)
                        YG2 = []
                        for i in range(len(arg_min)):
                            YG2.append(e[arg_min[i] - 1])
                        YG2_a = np.array(YG2)
                        target_scores = np.random.rand(len(arg_min))
                        flattened_inputs = YG2_a.reshape(len(arg_min), -1)
                        X_train, X_val, y_train, y_val = train_test_split(flattened_inputs, target_scores, test_size=0.2,
                                                                          random_state=42)
                        # 使用SSA优化RVFL超参数
                        optimal_params, optimal_val_loss = chimp_optimization(X_train, y_train, X_val, y_val)
                        # 使用最优参数训练最终模型
                        final_model = RVFL(n_hidden=int(optimal_params[0]), alpha=optimal_params[1],
                                           input_scaling=optimal_params[2])
                        final_model.fit(flattened_inputs, target_scores)
                        # 计算决策者权重
                        expert0 = calculate_decision_maker_weights(final_model, flattened_inputs, target_scores)
                        y2=np.zeros((5,7))
                        # 计算决策者权重
                        for i0 in range(len(arg_min)):
                            y2 += matrices3[i0] * expert0[i0]
                        Y2 = y2
            CMRS =[]
            for k in arg_min:
                CMRS.append(consensus_level(e[k - 1], Y2))
            # print("gongshidu")
            print(CMRS)
            # print(expert0)
            CCAr1=0
            c=[]
            for i in range(len(arg_min)):
                value = expert0[i]
                CCAr1+= CMRS[i]* value
                Group_weight[i+1]=value
            t += 1
    print('第二阶段各组权重：')
    print(Group_weight)
    print(len(Group_weight))
    print('最后共识度:')
    print(CCAr1)
    z=np.zeros((5,7))
    print("各小组领导者最终评价矩阵")
    for i in range(len(arg_min)):
        print(arg_min[i])
        print(e[arg_min[i]-1])
        z+=expert0[i]*e[arg_min[i]-1]
    print("总体矩阵")
    print(z)
    average_j=np.mean(z, axis=0)
    z1=np.zeros((5,7))
    for i in range(5):
        for j in range(7):
            z1[i][j]=z[i][j]-average_j[j]
    # print("损益函数")
    for i in range(5):
        for j in range(7):
            if z1[i][j]>=0:
                z1[i][j]=z1[i][j]**alpha
            else:
                z1[i][j]=-lamda_val2*((-z1[i][j])**beta)
    # print("价值函数：")
    z1=z1/7
    # print("前景值矩阵：")
    max_value = np.max(z1)
    min_value = np.min(z1)
    R1=np.zeros((5,7))
    G1=np.zeros((5,7))
    for i in range(5):
        for j in range(7):
            R1[i][j]=1-math.exp(lambda_val*abs((z1[i][j]-max_value)/(max_value-min_value)))
    for i in range(5):
        for j in range(7):
            G1[i][j]=1-math.exp(-lambda_val*abs((z1[i][j]-max_value)/(max_value-min_value)))
    Z=[]
    for i in range(5):
        s=0
        for j in range(7):
           s+=R1[i][j]+G1[i][j]
        Z.append(s)
    element=[]
    for i in range(5):
            element.append((Z[i], i))
    # 对元素进行从大到小排序
    element.sort(key=lambda x: x[0], reverse=True)
    # 只保留排序后的原始位置
    sorted_positions = [position+1 for value, position in element]
    print(sorted_positions)
    print(d)
    print(dd)
    print(ddd)
    print(len(GA))
    print(len(DM_number))
    total1=0
    for i in range(90):
        total1+=np.sum(np.abs(e[i]-xinmatrices[i]))
    print(total1/90)
    return Z
if __name__ == "__main__":
    CRP_algorithm()