pyenv install 3.11

pip3 install "gymnasium[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple

pip3 install "gymnasium[accept-rom-license]" -i https://pypi.tuna.tsinghua.edu.cn/simple


### algo
Actor-Critic
0.对每条轨迹，实际累计收益大于critic期望的时候加大动作概率，反之减少。似乎可以理解为比当前评估器好的方法加强概率、差的降低概率
1.举例来说，对某场游戏来说，假如最终赢了，那么认为这局游戏中每一步都是好的，如果输了，那么认为都是不好的。
2.策略梯度：每条轨迹，log动作概率乘以累计回报的期望。只对动作概率取log是为了简化计算。
3.动作好坏判断：针对一条轨迹，每个动作的后续实际奖励累加-critic预估的平均奖励，大于0说明这次动作是好的，小于0说明是差的

