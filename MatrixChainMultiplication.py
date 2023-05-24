import sys
import time
import random
import statistics
import matplotlib.pyplot as plt

#使用Brute-Force演算法的方法
def bruteForce_MCM(P):
    # 遞迴列舉矩陣i~矩陣j所有可能的括號化方式並計算在該括號方式下的最小純量乘法數
    def enumerate_parenthesizations(i, j):
        if i == j:  # 單獨一個矩陣，不需要括號
            return 0, 'A' + str(i)    #直接回傳該矩陣i，又因沒有進行乘法所以乘法數為0
        #兩參數用以紀錄當前分割下的最佳解
        min_multiplications = sys.maxsize   #初始設為無限(一個極大值)
        optimal_parenthesis = ""
        for k in range(i, j):
            # 分割矩陣鏈，遞迴求解左右兩部分的最佳括號方式
            left = enumerate_parenthesizations(i, k)
            right = enumerate_parenthesizations(k + 1, j)
            # 計算此括號化方式下的純量乘法次數
            multiplications = P[i - 1] * P[k] * P[j]
            # 更新最小的純量乘法次數以及對應的括號化方式
            if multiplications + left[0] + right[0] < min_multiplications:
                min_multiplications = multiplications + left[0] + right[0]
                optimal_parenthesis = f"({left[1]} x {right[1]})"
        #回傳最佳解
        return min_multiplications, optimal_parenthesis

    n = len(P) - 1  # Matrix-Chain中所含矩陣數量
    #求所有矩陣，即矩陣1~矩陣n的最佳分割方式以及其最小純量乘法數
    min_scalar_multiplications, optimal_parenthesization = enumerate_parenthesizations(1, n)
    return min_scalar_multiplications, optimal_parenthesization

#使用Dynamic Programming演算法的方法
def dynamicProgramming_MCM(P):
    n = len(P) - 1  # 矩陣的數量
    # 建立兩個二維陣列: m, s，大小皆為n*n
    #為了最大化矩陣利用，index從0開始，即矩陣i~矩陣j鏈乘法結果記在m[i-1][j-1](s[i-1][j-1])中
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    # 初始化對角線元素
    for i in range(n):
        m[i][i] = 0
    # 計算最佳括號(切割)方式
    for l in range(2, n + 1):  # 每個可能的鏈長度，從2開始(for 2 to n)
        for i in range(1, n - l + 2): #每個起始指標i(for 1 to n-l+1)
            j = i + l - 1   #每個i有其對應的結束指標j
            m[i-1][j-1] = sys.maxsize   #初始值為無限(or一個極大值)
            #矩陣i~矩陣j鏈乘法下的不同括號方式
            for k in range(i, j):
                # 計算最小純量乘法次數
                temp_cost = m[i-1][k-1] + m[k][j-1] + P[i-1] * P[k] * P[j]
                #取最小值
                if temp_cost < m[i-1][j-1]:
                    m[i-1][j-1] = temp_cost
                    s[i-1][j-1] = k

    # 建構最佳括號方式的方法(採用遞迴式)
    def construct_parenthesization(i, j):
        if i == j:
            return f"A{i}"
        else:
            k = s[i-1][j-1]
            left = construct_parenthesization(i, k)
            right = construct_parenthesization(k+1, j)
            return f"({left} x {right})"

    #求所有矩陣，即矩陣1~矩陣n的最佳分割方式以及其最小純量乘法數，m[0][n-1]中結果即是
    min_scalar_multiplications = m[0][n-1]
    #利用陣列s中內容得到最佳括號方式
    optimal_parenthesization = construct_parenthesization(1, n)
    return min_scalar_multiplications, optimal_parenthesization

#產生隨機大小為n之P陣列的方法
def generate_random_P(n):
    P = []
    for i in range(n+1):
        # 生成 5 的倍數作為 P 的元素
        element = random.randint(1, 20) * 5
        P.append(element)
    return P

#為了比較兩演算法所需時間以及不同n下的影響，主程式編寫邏輯如下：
#1. n由5開始直到15各產生五組P
#2. 對於每個n的五組P皆分別套用暴力演算法及動態規劃演算法
#3. 對於五組P執行的時間結果取平均，作為兩演算法各自在該n下的平均執行時間
#4. 以x軸為n，y軸為執行時間繪製折線圖
N = []
BF_MET = []
DP_MET = []
for n in range(5,16):
    BF_execution_time = []
    DP_execution_time = []
    for i in range(5):
        P = generate_random_P(n)
        start_time = time.time()
        BF_result = bruteForce_MCM(P)
        BF_execution_time.append(time.time() - start_time)
        # print("Minimum scalar multiplications:", BF_result[0])
        # print("Optimal parenthesization:", BF_result[1])

        start_time = time.time()
        DP_result = dynamicProgramming_MCM(P)
        DP_execution_time.append(time.time() - start_time)
        # print("Minimum scalar multiplications:", DP_result[0])
        # print("Optimal parenthesization:", DP_result[1])
    N.append(n)
    BF_MET.append(statistics.mean(BF_execution_time))
    DP_MET.append(statistics.mean(DP_execution_time))

print(BF_MET)
print(DP_MET)
#利用折線圖視覺化結果
plt.plot(N, BF_MET, label='Brute Force')
plt.plot(N, DP_MET, label='Dynamic Programming')
plt.xlabel('n')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time of Matrix-Chain Multiplication')
plt.legend()
plt.show()
