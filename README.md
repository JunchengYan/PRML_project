# PRML_project
###### 1. 代码各部分的作用和使用

代码部分中main.py的作用是进行svm、resnet模型的训练和测试、评估，代码中各函数含义清晰。其中设置了三个args，分别是实验名exp_name, 模型名model以及数据集dataset，运行python时请指定参数。

代码部分中data.py的作用是加载数据集，其中load_and_split_data用于正常加载公开数据集，load_Tsinghua_data用于加载清华数据集全集，load_scene_Tsinghua_data用于加载不同的scene的清华数据集，StairDataset为继承的Dataset类。

代码部分中model.py是Resnet模型的实现

代码部分中refine.py是将部分清华数据集补充进公开数据集并进行训练和测试的代码。cal_acc函数为分scene计算模型在清华数据集上的正确率，单独运行refine文件时在args中指定模型即可。主函数其余部分为对增补数据集后的模型进行测试的代码，仍然在args中指定模型。

代码部分中GradCAM.py为Graf-CAM算法的实现文件，运行可以查看结果，将img_path改为需要的图片即可。

代码部分中distill.py为模型蒸馏部分代码，只需要指定实验名称运行即可。

代码部分中utils.py为存储io工具的部分，方便对实验过程记录。



其余文件中，json文件夹为存储数据集划分文件的文件夹，requirements.txt为代码依赖库的文件。

logs为此前实验的实验记录以及一些最佳模型的记录，其中的run.log均为运行时的记录可以查看，下面分条意义叙述如下：

####

**神经网络：**

Res_public为基本公开数据集上的结果，**其中models内的model.t7为resnet最佳模型**

Res_tsinghua为在基本清华数据集上测试的结果

Res_finetune_test为在清华数据集上分scene测试的结果

Res_finetune_exp为在增补数据集后，训练、测试的结果，内含重新训练好的模型

Res_finetune_check为改进后重新在清华数据集上分scene测试的结果

####

*以svm为前缀的5个文件夹内涵和上述一致，但是svm**不需要保存最佳模型***



res50为老师模型finetune的训练结果，内含最佳resnet50的模型

student_model1为学生模型单独训练和测试的结果，内含模型

distill_exp4为蒸馏训练和测试的结果，内含模型

distill_result为蒸馏后的测试结果





###### 2. 代码运行环境和库版本

代码运行环境为：python==3.8.16	CUDA Version: 12.0  

各外部库版本为：

matplotlib==3.7.1

numpy==1.24.3

opencv_python==4.7.0.72

Pillow==9.5.0

scikit_learn==1.2.2

torch==1.11.0+cu113

torchstat==0.0.7

torchvision==0.12.0+cu113

tqdm==4.65.0

注意pytorch和torchvision应当通过torch官方网站下载，直接用pip安装requirements.txt会失败





###### 3. 数据集存放位置

**我的训练中，数据集名称为stairs，分为public和tsinghua两个文件夹。stairs就位于整个项目codes文件夹下，直接放置进去运行就行。**





###### 4. 复现你的最佳模型的训练过程命令以及对应地在测试集上测试的命令

**神经网络：**

（自动会测试评估）训练：python main.py --exp_name [] --model resnet --dataset public

清华测试：python main.py --exp_name [] --model resnet --dataset tsinghua

**注：如果要测试上一步训练好的模型，请将tsinghua_main()最后一行395的test_eval()最后一个参数改为你刚才训练的最佳模型的logs地址，只需要改‘res_public’位置即可**

**SVM：**

（自动会测试评估）训练：python main.py --exp_name [] --model svm --dataset public

清华测试：python main.py --exp_name [] --model svm --dataset tsinghua



请注意：exp_name不要和logs中的文件重名。

也可以整体删除掉codes中的logs文件夹，重新开始训练（那样神经网络的清华测试就必须要改地址，因为我原本训练的最佳模型已经被删除）。
