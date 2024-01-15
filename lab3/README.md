### ML LAB3代码说明
1. my_model, pretrained_model是自定义CNN和基于预训练CNN的模型 (可能需要自行修改各层参数)
2. lab3_data_scratch 参考老师和示例代码，重新构造了分割dataset / dataloader的函数
3. train_eva 参考示例代码，执行训练预验证的逻辑
4. visual_history 使用实例代码，可视化运行过程——绘制图像
5. main 接收参数 训练，验证，保存模型的入口
    - --model_select 选择模型
    - --lr_select 选择参数调整策略
    - --pr_select 原本想做测试时加载哪一部分参数的，后放弃
    - --epoch 训练轮数
    - --lr 初始学习率
6. test_one_file 读取一个视频文件，加载模型进行预测
    - --model_select 选择模型
    - --pt_select 选择模型参数
    - --video_path 视频路径
7. pics 运行结果的图片
8. environment.yml packages.txt conda虚拟环境

**运行示例**
    
    python main.py --model_select pre_trained --lr_select re --epoch 50 --lr 1e-4  
    /*请将data文件夹放在py文件同级目录下*/
    python test_one_file.py --model_select pre_trained --pt_select acc --video_path ./data/Biking/v_Biking_g01_c01.avi

**我的代码仓库**
    https://github.com/AsVoider/VideoClassfication.git