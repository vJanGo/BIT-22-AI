import numpy as np

# 读取数据
def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)


points_3d = read_points('house.p3d')

# 设置相机参数
# 设置相机的内参数矩阵 K
fx = 1000  # 焦距在x方向
fy = 1000  # 焦距在y方向
u = 320   # 主点x坐标
v = 240   # 主点y坐标
# 设置输出格式
np.set_printoptions(precision=6, suppress=True)

K = np.array([[fx, 0, u],
              [0, fy, v],
              [0, 0, 1]])

# 设置外参数：旋转矩阵 R 和平移向量 T
R = np.array([[1, 3, 5],
              [0, 1, 6],
              [0, 0, 1]])  
T = np.array([1, 1, 1])  

#计算像素坐标

def project_points(points_3d, K, R, T):
    # 将点转到相机坐标系
    points_cam = (R @ points_3d.T).T + T
    
    # 计算投影
    points_2d = (K @ points_cam.T).T
    
    # 将齐次坐标转换为常规坐标
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    
    return points_2d[:, :2]

# 投影得到像素坐标
points_2d = project_points(points_3d, K, R, T)
# print("像素坐标：\n", points_2d)

def construct_projection_matrix(points_3d, points_2d):
    n = points_3d.shape[0]
    A = []
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P

# 计算投影矩阵 P
P = construct_projection_matrix(points_3d, points_2d)

def decompose_projection_matrix(P):
    # 分解 P = K[R|T]
    M = P[:, :3]  # 3x3矩阵

    # QR分解来分解旋转和内参矩阵
    K, R = np.linalg.qr(np.linalg.inv(M))
    
    # 通过调整 K 的符号使其成为上三角矩阵
    T = np.linalg.inv(K) @ P[:, 3]
    
    # 归一化K
    K /= K[2, 2]
    
    return K, R, T

# 分解得到 K, R, T
K_est, R_est, T_est = decompose_projection_matrix(P)

# 输出矩阵
def matrices(K_orig, R_orig, T_orig, K_new, R_new, T_new):
    
    
    # 内参矩阵 K
    print("原始内参矩阵 K:\n", K_orig)
    print("分解得到的内参矩阵 K:\n", K_new)
    

    # 旋转矩阵 R
    print("原始旋转矩阵 R:\n", R_orig)
    print("分解得到的旋转矩阵 R:\n", R_new)
   

    # 平移向量 T
    print("原始平移向量 T:\n", T_orig)
    print("分解得到的平移向量 T:\n", T_new)
   
matrices(K,R,T,K_est,R_est,T_est)

def reconstruct_projection_matrix(K, R, T):
    # 将 R 和 T 组合成 3x4 矩阵
    RT = np.hstack((R, T.reshape(3, 1)))
    
    # 计算 P = K * [R | T]
    P_reconstructed = K @ RT
    
    return P_reconstructed


# 构建复原后的投影矩阵
P_reconstructed = reconstruct_projection_matrix(K, R, T)

# 比较原始投影矩阵 P 和复原的投影矩阵 P_reconstructed
print("原始投影矩阵 P:\n", P)
print("复原的投影矩阵 P_reconstructed:\n", P_reconstructed)
