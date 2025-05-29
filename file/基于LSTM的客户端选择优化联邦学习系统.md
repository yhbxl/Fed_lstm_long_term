基于LSTM的客户端选择优化联邦学习系统

First Author1[0000-1111-2222-3333] and Second Author2[1111-2222-3333-4444]

1 College of Computer Science and Technology, Harbin Engineering

 University, Harbin 150001, China

 {2015080325, wangmengchao, baozhida, yyd

 20000614}@hrbeu.edu.cn,

 2 College of Computer Science and Technology,Harbin Engineering University,

 Harbin 150001, China



**Abstract.**传统的联邦学习（Federated Learning）采用随机客户端选择策略，未能充分考虑客户端性能差异和数据分布异构性对系统整体性能的影响。本文提出一种融合性能与数据多样性的客户端选择方法，综合考虑客户端性能和数据多样性，显著优化联邦学习系统的收敛速度和模型精度。实验证明，与传统的Fed-Avg随机选择方法相比，所提出的方法在收敛速度上提高了35%，在模型精度上提升了7.2%，通信成本减少了52%。该方法在各种非独立同分布（Non-IID）数据环境中表现出色，为资源受限场景下的联邦学习提供了有效解决方案。

**Keywords:**联邦学习，客户端选择，LSTM模型，混合策略，性能优化.

1      引言

联邦学习作为一种分布式机器学习范式，允许多个参与方在保护本地数据隐私的前提下协作训练共享模型。然而，传统联邦学习中的联邦平均（Fed-Avg）算法采用随机客户端选择策略，没有充分考虑客户端异构性和数据分布差异对系统性能的影响。随着边缘设备数量的增加和计算资源的不平衡分布，提高客户端选择效率已成为优化联邦学习系统的关键。

当前联邦学习中的客户端选择策略主要集中于客户端的计算资源、通信带宽、本地训练损失或对全局模型的贡献度等方面。McMahan等人[1]提出的原始Fed-Avg算法采用随机选择策略，难以适应异构环境。Nishio等人[2]提出考虑客户端计算能力的选择方法，但未充分利用客户端历史性能信息。Cho等人[3]采用基于贡献度的策略，但忽视了数据分布多样性对全局模型的影响。

为解决上述问题，本文提出一种基于历史性能表现的混合客户端选择策略，融合性能指标与数据多样性，创新性地引入LSTM模型预测客户端未来表现，提升选择准确率。主要贡献如下：

(1) 设计融合性能与多样性的评分机制，提升选择公平性与代表性；

(2) 引入历史窗口机制记录客户端性能轨迹，实现动态客户端评估；

(3) 构建LSTM预测模型，辅助实现智能化、自适应的客户端筛选。

实验结果表明，所提出的方法在模型精度、收敛速度和通信效率上均优于现有方法。



2      相关工作

自McMahan等人[1]提出联邦学习以来，客户端选择策略始终是影响系统性能的关键因素之一。传统FedAvg算法采用随机选择，虽然实现简单，但忽略了客户端性能的差异性。Wang等人[4]指出，客户端的计算能力直接影响系统效率，提出考虑异构资源的选择策略。Nishio等人[2]提出FedCS算法，根据计算与通信能力筛选客户端；Chai等人[5]进一步考虑网络带宽因素，降低聚合延迟。这些策略提升了系统资源利用率，但未将客户端对模型性能的贡献纳入考量。

+++++++++++++



2.1    性能导向的客户端选择



性能导向策略关注客户端对模型精度的提升效果。Cho等人[3] 基于历史训练损失选择潜力大的客户端。Li等人[6] 基于梯度变化设计选择策略，优先选择梯度动态大的客户端。虽然此类方法提高了模型精度，但未涉及客户端数据的多样性。

2.2    数据多样性导向的客户端选择

数据多样性是联邦学习中提升全局模型泛化性能的关键。Fraboni等人[7]研究表明，选择具有多样化数据分布的客户端可提高模型在非同分布环境下的泛化能力，缓解非IID带来的性能损失。Zhang等人[8]提出基于核心集构建的多样性感知选择策略，有效缓解数据偏差问题。但这些方法未结合客户端性能因素，导致收敛效率低下。

综上，本文提出的混合策略融合了性能和多样性两个维度，通过可调权重机制灵活平衡不同场景需求，克服了现有方法的局限性。

3      方法

3.1    系统架构

如图1所示本文系统包含四个主要模块：(1)联邦学习训练器，负责负责初始化并聚合全局模型；(2)客户端选择器，实现有策略的客户端选择；(3)性能评估器，记录和分析客户端历史表现；(4)LSTM预测模块，进行序列预测，辅助选择。

系统工作流程如下：服务器初始化全局模型并发布训练任务；客户端选择器根据评分机制选择合适的客户端参与训练；选中的客户端执行本地训练并上传更新；服务器聚合后生成新一轮模型。

3.2    混合策略客户端选择

混合策略客户端选择方法综合考虑客户端性能和数据多样性，定义客户端评分函数如下：



Si表示客户端i的综合得分;

Pi为客户端最新一轮训练精度(![img]();

Di为历史精度方差(![img](),用于度量数据的多样性;

![img]()为可调权重系数;

β是![img]()放大因子，增强方差的影响

我们采用滑动窗口机制存储每个客户端近 k 轮训练精度，并据此计算评分 ![img]()，最终选择得分前 n 名的客户端参与训练。

3.3    LSTM预测模块

我们基于客户端最近 k 轮精度记录构建输入序列，通过双层LSTM网络进行训练，输出下一轮预测精度作为性能得分参考。LSTM模型结构如图2所示，包括三个主要部分： 

输入层：滑动窗口长度为 k 的精度序列；

LSTM层：2层，128维隐藏单元；

输出层：全连接层，输出预测得分。

LSTM模型定义如下：

模型可处理多种输入形状，适应不同类型的数据，通过多层LSTM结构捕获数据的长期依赖关系，提高预测准确性。

4      实验

4.1    实验设置

数据集：MNIST和CIFAR-10，使用Dirichlet(α=0.5)划分模拟非IID情况;

模型配置：LSTM输入维度28，层数2，Dropout率0.2，类别数10;

客户端数量：客户端总数为100，每轮选择5-20个客户端，本地训练轮次为5，联邦总轮数为10;

全局训练轮次为10。客户端选择策略包括随机选择、基于性能选择、基于多样性选择和混合策略选择。

对比策略：随机选择、性能优先、多样性优先、混合策略；

评估指标： 模型精度、收敛速度、通信成本、鲁棒性。



4.2    策略性能对比

图2展示了四种客户端选择策略在MNIST数据集上的性能比较。

实验结果表明，混合策略在模型精度和收敛速度上均优于其他方法。在MNIST数据集上，混合策略在10轮后测试精度达93.5%，相较于随机策略提高7.2%，相较于性能导向与多样性导向分别提高3.1%和4.5%。

在收敛速度方面，混合策略在第5轮达到90%精度，比随机选择提前3轮，表明混合策略可显著加速模型收敛。

这主要得益于混合策略同时考虑了性能好的客户端和数据分布多样的客户端，平衡了快速收敛和模型泛化能力。





4.3    通信效率

图3展示了不同客户端选择策略的通信成本对比。混合策略通过选择更有价值的客户端，实现了更高效的通信。为达到90%精度目标，混合策略所需通信量仅32MB数据，较随机选择减少了52%，大幅降低了系统通信开销。当Dirichlet α值降至0.1时（强非IID），混合策略精度仍达85.7%，较随机策略高13.4%。





4.4    鲁棒性分析

表1展示了不同非IID程度（通过Dirichlet参数α=0.1控制）下各策略的性能表现。结果表明，随着非IID程度增加（α值减小），所有方法性能都有所下降，但混合策略表现出更强的鲁棒性。在极端非IID环境（α=0.1）下，混合策略仍保持85.7%的精度，比随机选择高13.4%，证明了混合策略在异构环境中的适应性。





5      结论

本文提出融合客户端历史性能与数据多样性的混合客户端选择策略，引入LSTM预测模型，实现更精准的选择决策。实验结果表明，与传统方法相比，该方法在模型精度、收敛速度和通信效率上均有显著提升。在非IID环境中展现出优越的鲁棒性，为资源受限的联邦学习场景提供了有效的解决方案。

未来工作将探索动态调整权重参数的自适应方法，进一步优化客户端选择策略，并拓展至更复杂的多任务学习与大规模设备网络中。

参考文献

[1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.

[2] Nishio T, Yonetani R. Client selection for federated learning with heterogeneous resources in mobile edge[C]//ICC 2019-2019 IEEE International Conference on Communications (ICC). IEEE, 2019: 1-7.

[3] Cho Y J, Wang J, Joshi G. Client selection in federated learning: Convergence analysis and power-of-choice selection strategies[J]. arXiv preprint arXiv:2010.01243, 2020.

[4] Wang S, Tuor T, Salonidis T, et al. Adaptive federated learning in resource constrained edge computing systems[J]. IEEE Journal on Selected Areas in Communications, 2019, 37(6): 1205-1221.

[5] Chai Z, Ali A, Zawad S, et al. TiFL: A tier-based federated learning system[C]//Proceedings of the 29th International Symposium on High-Performance Parallel and Distributed Computing. 2020: 125-136.

[6] Li T, Sahu A K, Zaheer M, et al. Federated optimization in heterogeneous networks[J]. Proceedings of Machine Learning and Systems, 2020, 2: 429-450.

[7] Fraboni Y, Vidal R, Kameni L, et al. Clustered sampling: Low-variance and improved representativity for clients selection in federated learning[C]//International Conference on Machine Learning. PMLR, 2021: 3407-3416.

[8] Zhang C, Xie Y, Bai H, et al. A fair and efficient federated learning framework using coreset[J]. IEEE Transactions on Neural Networks and Learning Systems, 2022.