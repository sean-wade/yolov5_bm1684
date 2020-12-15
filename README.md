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
    bmnet/bmnetd/bmnetd --model=YJH/D200/D200.cfg --weight=YJH/D200/D200.weights --net_name=D200 --outdir=./YJH/D200/Bmodel/ --shapes=[1,3,608,608] --target=BM1684
    
    生成UModel
    bmnet/bmnetd/bmnetd --mode=GenUmodel --model=YJH/D200/D200.cfg --weight=YJH/D200/D200.weights --net_name=D200 --outdir=./YJH/D200/Bmodel/
    
    bmnet/bmnetd/bmnetd --model=YJH/edge/yolo.cfg --weight=YJH/edge/yolo.weights --net_name=edge --outdir=./YJH/edge/Bmodel/ --target=BM1684
	
5、测试
	bmrt_test --bmodel YJH/cabinet/Bmodel/compilation.bmodel --loopnum 10
