from PIL import Image
import feature
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
import sklearn.tree

if __name__ == "__main__":

    face_dir="./datasets/original/face/face_"
    nonface_dir="./datasets/original/nonface/nonface_"

        #取出face的图片特征
    for i in range(500):
        dir=face_dir+str("%.3d"%i)+".jpg"
        im=Image.open(dir).convert("L").resize((24,24))
        NPD = feature.NPDFeature(np.array(im))
        if(i==0):
            X=NPD.extract()
            y=np.array([1])
        else:
            X=np.vstack((X,NPD.extract()))
            y=np.vstack((y,np.array([1])))

        #取出nonface的图片特征
    for i in range(500):
        dir=nonface_dir+str("%.3d"%i)+".jpg"
        im=Image.open(dir).convert("L").resize((24,24))
        NPD = feature.NPDFeature(np.array(im))
        X=np.vstack((X,NPD.extract()))
        y=np.vstack((y,np.array([-1])))

    cache_file_dir="X_data.pkl"
    output=open(cache_file_dir,"wb")
    pickle.dump(X,output)
    output.close()

    cache_file_dir="y_data.pkl"
    output=open(cache_file_dir,"wb")
    pickle.dump(y,output)
    output.close()

        #print(X)
        #NPD=feature.NPDFeature(np.array(im))
        #X.append(np.mat(NPD.extract()))

    # write your code here

    cache_file_dir = "y_data.pkl"       #数据文件
    input = open(cache_file_dir, "rb")
    y = pickle.load(input)
    input.close()

    cache_file_dir = "x_data.pkl"
    input = open(cache_file_dir, "rb")
    X = pickle.load(input)
    input.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    for depth in range(2,6):
        for number_tree in range(1,6):
            classifier = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier, number_tree,depth)
            classifier.fit(X_train, y_train)
            print("max_depth:",depth," number of tree:",number_tree)
            print(np.mean(classifier.predict(X_test) == np.transpose(y_test)))



    pass




