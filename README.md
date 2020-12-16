# yolov5_bm1684
bitmain bm1684 inference code of YoloV5...


# 转换模型：
1、找个服务器load docker

    docker load -i bmnnsdk2-bm1684_v2.0.1.docker
    
    
2、运行docker

    tar zxvf bmnnsdk2-bm1684_v2.0.1.tar.gz（实际上现在已经是v2.1.0了）
    cd bmnnsdk2-bm1684_v2.0.1
    ./docker_run_bmnnsdk.sh
    

3、docker里配置环境

    cd /workspace/scripts/
    ./install_lib.sh nntc
    
    cd /workspace/scripts/
    source envsetup_cmodel.sh


4、转换
    
    下面几行是转darknet框架的yolov3
    bmnet/bmnetd/bmnetd --model=YJH/D200/D200.cfg --weight=YJH/D200/D200.weights --net_name=D200 --outdir=./YJH/D200/Bmodel/ --shapes=[1,3,608,608] --target=BM1684
    
    生成BModel
    bmnet/bmnetd/bmnetd --mode=GenUmodel --model=YJH/D200/D200.cfg --weight=YJH/D200/D200.weights --net_name=D200 --outdir=./YJH/D200/Bmodel/
    
    bmnet/bmnetd/bmnetd --model=YJH/edge/yolo.cfg --weight=YJH/edge/yolo.weights --net_name=edge --outdir=./YJH/edge/Bmodel/ --target=BM1684
    
    下面一行是转pytorch下的yolov5
    python3 -m bmnetp --model=zhanghao/yw_5s.torchscript.pt --shape=[1,3,640,640] --net_name="yw_5s" --outdir=zhanghao/yw_5s --target=BM1684
	
5、测试
	bmrt_test --bmodel YJH/cabinet/Bmodel/compilation.bmodel --loopnum 10


# 代码编译：

1、编译

    进入docker环境后
    sudo ./docker_run_bmnnsdk.sh
    当前实际目录会映射到容器中的/workspace下面，在容器中进入该代码目录
    cd yolov5_bm1684
    make
    会在当前目录下生成：
    v5_test
    
2、测试

    将 v5_test 拷贝到 BM1684设备上
    执行
    ./v5_test ./test_jpgs/ ./yw_5s.bmodel