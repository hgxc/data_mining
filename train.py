# -- coding:utf-8 --
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt
import numpy as np
import  os
import itertools

def load_data(filename):
    '''读取单个文件夹中的字符信息'''
    with open(filename,'r') as file:
        lines = file.readlines()
        dataset = [[]for i in range(len(lines))]
        #print('f',dataset)
        for i in range(len(dataset)):
            dataset[i][:]=(item for item in lines[i].strip().split(' '))
        print(len(dataset))
        #print(dataset)

        return dataset

def create_VSM(num):
    file_path='./SData/'
    file_class=['Training-800/Crude','Training-800/Grain','Training-800/Interest','Training-800/Trade','Testing-40']
    file_count = 0      #计算文件个数
    #test_count = 0

    total_data=[]
    lost_data=[]        #丢失的数据
    words = []      #VSM的列
    y_data=[]

    for i,x in enumerate(file_class):
        file_name = os.listdir(file_path+str(x))
        file_name.sort(key=lambda x:int(x[:-4]))
        for file in file_name:
            print(file)
            if os.path.getsize(file_path+str(x)+'/'+file)==0:
                lost_data.append((x,file))
                continue
            if x == file_class[0]: #对训练集设置标签
                y_data.append(0)
                file_count+=1
            elif x == file_class[1]:
                y_data.append(1)
                file_count+=1
            elif x == file_class[2]:
                y_data.append(2)
                file_count+=1
            elif x == file_class[3]:
                y_data.append(3)
                file_count+=1
            elif x == file_class[4]:
                file_count+=1
            temp=load_data(file_path+str(x)+'/'+file)
            temp = np.asarray(temp)
            total_data.append(temp)
            word=list(temp[:,0])
            words=list(set(words).union(set(word)))
            print(x)


    words = sorted(words)
    total_data=np.asarray(total_data)

    #向量空间模型
    VSM = np.zeros((file_count,len(words)))


    print(VSM.shape)
    #填充VSM值
    for i in range(VSM.shape[0]):
        index = [words.index(total_data[i][j,0]) for j in range(len(total_data[i]))]
        VSM[i][index] = total_data[i][:,1]

    y_data = np.asarray(y_data)
    print(VSM)
    print(lost_data)

    return VSM,y_data

def network_part(x_data,y_data):


    # 将打乱后的数据集分割为训练集和验证集和测试集，
    X_train = x_data[:-40]
    Y_train = y_data

    x_test = x_data[-40:]       #测试集


    # 随机打乱数据（因为原始数据是顺序的，顺序不打乱会成的随机数都一样
    np.random.seed(66)  # 使用相同的seed，保证输入特影响准确率
    # seed: 随机数种子，是一个整数，当设置之后，每次生征和标签一一对应
    np.random.shuffle(X_train)
    np.random.seed(66)
    np.random.shuffle(Y_train)
    tf.random.set_seed(66)

    x_train = X_train[:-40]     #训练集
    y_train = Y_train[:-40]


    x_verification = X_train[-40:]      #验证集
    y_verification = Y_train[-40:]

    print('x_train',x_train.shape)

    print('y_train',y_train.shape)


    y_train= tf.one_hot(y_train, depth=4)
    y_verification = tf.one_hot(y_verification,depth=4)
    print(y_train)


    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(128,activation='relu'),
        layers.Dense(4, activation='softmax')]
    )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.1),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])



    model.fit(x_train,y_train,epochs=100,batch_size=32)
    model.summary()

    model.evaluate(x_verification,y_verification)


    #测试集结果
    predict=model.predict(x_test)
    result = np.argmax(predict,axis=1)
    print(result)





if __name__ == '__main__':
    num = 4     #代指文档数
    VSM, y_data= create_VSM(num)

    tfidf = TfidfTransformer().fit_transform(VSM)
    print(tfidf)
    tfidf=tfidf.toarray()
    print('tfidf',tfidf.shape)

    network_part(tfidf,y_data)





