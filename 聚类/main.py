import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

import matplotlib.pyplot as plt

def visualization(img, cluster_set, k):
    #初始化输出矩阵
    show_image = np.empty((k, 5), dtype=int)
    #在每个聚类结果中随机抽取5个样本下标
    for i in range(k):
        temp = np.array(np.where(cluster_set == i), dtype=int)
        temp = temp[0]
        show_image[i] = np.random.choice(temp, 5, replace=False)
        
    show_image = show_image.T

    fig = plt.figure()
    #初始化输出格式
    ax = fig.subplots(5, k)
    for i in range(5):
        for j in range(k):
            if i == 0:
                ax[i, j].set_title("%d" % (j))
            ax[i][j].imshow(img[show_image[i, j]])
            ax[i][j].axis('off')
    fig.show()


 # 图像个数
print('读入图片')
path = 'cifar-10-batches-py/'  # 路径可以自己定义
data_label = np.zeros(10000)   #创建标签数组
data_array = np.zeros((10000, 3072)) #创建图片数组
cur_dict = unpickle(path + 'data_batch_1') #输入文件路径
for i in range(10000):
    data_array[i] = cur_dict[b'data'][i][:]   #依次输入每张图片像素点
    data_label[i] = cur_dict[b'labels'][i]    #输入图片标签
data_arraycal = data_array   
#把10000*3072的矩阵从新归化为10000*32*32*3                 
data_array = data_array.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
data_label = data_label.astype('float32')
print('done')





# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离
    #return np.sum(abs(x - y))


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i,:] = np.array(dataSet[index,:])
    return centroids


# k均值聚类


def KMeans(dataSet, k):
    m = np.shape(dataSet)[0] # 行的数目
    print(m)
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True
    time = 0

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False
        time = time + 1

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :]) 

                
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    print(time)
    return centroids, clusterAssment


#聚类个数
k = 8
#聚类
centroids, clusterAssment = KMeans(data_arraycal, k)

#计算每个聚类所有点间的平均距离返回一个一维数组
def avgc(clusterAssment, data_arraycal, k, avgclist):
    for i in range(k):
        avg =0   #记录一个聚类内所有点的距离之和
        n = 0    #记录所有点与点个数和
        for j in range(10000):
            if clusterAssment[j, 0] == i:
                for k in range(j+1, 10000):
                    if clusterAssment[k, 0] == i:
                        #把所有点的聚类累加
                        avg += distEclud(data_arraycal[j, :], data_arraycal[k, :])
                        n += 1
        #求得点间平均距离
        if n == 0:
            avgclist[i] =0
        else:
            avgclist[i] = avg / n
    
    return avgclist
    
                        
avgclist = np.zeros(k)  #初始化存储点间平均距离的数组
avgclist = avgc(clusterAssment, data_arraycal, k ,avgclist) #计算求得平均距离数组


#求dbi的值                       
def dbi(centroids, avgclist, k):
    dbi_sum = 0
    maxi = 0
    for i in range(k):
        for j in range(i+1 ,k):
            dcen = (avgclist[i] + avgclist[j])/distEclud(centroids[i], centroids[j])
            if dcen > maxi:
                maxi = dcen
        dbi_sum += maxi
    return dbi_sum / k
db = dbi(centroids, avgclist, k)
print('DBI = ',db)

    
data_array_int = data_array.astype(int)  
visualization(data_array_int, data_label, k) #用标签在每类中选取五个图片输出

visualization(data_array_int, clusterAssment, k) #用聚类标签在每个聚类中选取五个图片输出

            
    
        








