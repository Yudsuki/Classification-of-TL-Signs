import os
import shutil
baseDir="./data/"
for i in os.listdir(baseDir+"train"):
    if(os.path.isdir(baseDir+"train/"+i)):
        files=os.listdir(baseDir+"train/"+i)
        train=files[:int(len(files)*0.7)]
        test=files[int(len(files)*0.7):]
        print(i)
        print(len(files))
        print(len(test))
        print(len(train))
        for name in test:
            if(os.path.exists(baseDir+"test/"+i)):
                shutil.move(baseDir+"train/"+i+'/'+name,baseDir+"test/"+i)
            else:
                os.makedirs(baseDir+"test/"+i)
                shutil.move(baseDir + "train/" + i + '/' + name, baseDir + "test/" + i)
