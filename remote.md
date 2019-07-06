######远程打开代码
ssh -R 52698:127.0.0.1:52698 daiyuchao_UG@10.69.35.132
rmate -p 52698 filename
scp pytorch-cifar-master.tar.gz daiyuchao_UG@10.69.35.132:/test/      #无法传输文件夹


CUDA_VISIBLE_DEVICES=1 python2 main.py #指定gpu运行脚本

输出重定向
```
f_handler=open('out.log', 'w')
sys.stdout=f_handler
print 'hello' 
```
######tensorborad远程访问
ssh -L 16006:127.0.0.1:6006 daiyuchao_UG@10.69.35.132
tensorboard --logdir="/path/to/log-directory"
访问http://127.0.0.1:16006/

######从服务器下载目录
```
scp -r daiyuchao_UG@10.69.35.132:~/test/pytorch-cifar/infos ~/test/dl_homework/logs/ 
```



> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->；