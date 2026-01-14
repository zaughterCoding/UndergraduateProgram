## 运行说明

### 新建conda环境
> cd ccl_11

> conda create -n ccl11 python=3.10

> conda activate ccl11

### 安装依赖，以及fa2
> pip install -r requirements.txt

### 下载基础模型(需要确保git-lfs已安装)
> git clone https://www.modelscope.cn/Qwen/Qwen2.5-VL-7B-Instruct.git
请确保下载后的Qwen2.5-VL-7B-Instruct文件夹与train_eqcc1.py处于同级目录。

### 运行推理

首先需要将我们提供的lora权重，对第266行进行注释，将268-273行代码取消注释，并加载我们的Lora权重路径到第268行。
子任务一Lora权重文件：03-23-21-27task1_eval_{'macro'_ {'P'_ 0.9557, 'R'_ 0.9531, 'F1'_ 0.954}, 'micro'_ {'P'_ 0.9533, 'R'_ 0.9533, 'F1'_ 0.9533}}
子任务二Lora权重文件：04-08-23-02task2_eval_{'ROUGE-1'_ 0.7209, 'ROUGE-2'_ 0.6118, 'ROUGE-L'_ 0.6914, 'score'_ 0.6764}
并将第706行设置为我们提供的测试集路径，子任务一测试集为task1_test.json，子任务二测试集为task2_test.json，注意需要依次替换路径每次只能训练推理一个子任务！！！
将第902行代码注释，取消第903行的注释
然后运行脚本：
> python train_eqcc1.py

这会加载我们提供的lora权重，若想要加载自行训练后的lora权重，将第268行替换为对应的lora路径即可。并在在测试集上进行推理并生成最终结果到指定文件task1-answer-submit.json以及task2-answer-submit.json

### 运行训练
将train_eqcc1.py文件中62行中的default设置为task1来对子任务一进行训练和推理，指定为task2来对子任务二进行训练和推理，通过指定default分别单独对两个任务进行训练
71和72行为两个任务的训练集，第77行为模型路径，可根据需要进行修改。修改后，在命令行或终端中，进入脚本所在的目录，
>  python train_eqcc1.py

在训练过程中会自动保存epoch检查点和较优检查点到lora文件夹下。