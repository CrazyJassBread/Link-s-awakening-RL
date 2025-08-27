## 项目目标
本项目尝试训练一个能解决Zelda简单问题的agent，尝试使用纯PPO来进行探索，计划在之后引入多模态LLM进行规律总结和子任务划定，希望能通过结合两者取的更好的效果

目标是让agent在人工标注数据的引导下可以解决一号迷宫的大部分谜题

一号迷宫缩略图如下所示![迷宫缩略图](image\迷宫一.png)

## 框架介绍
`gamestate` 文件夹中存放有gb原始文件信息
`play the game` 文件夹中可以人工游玩游戏
`RL` 文件夹中是目前正在尝试的强化学习方法

## 任务进展安排
- [x] 尝试人工通关游戏
- [ ] 选择简单的迷宫房间用PPO训练agent
- [ ] 使用LLM总结游戏规则和子任务划分

## 参考资料、项目

游戏内存信息：[The Legend of Zelda: Link's Awakening (Game Boy) - Data Crystal](https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy))

GameBoy模拟器：[PyBoy](https://github.com/Baekalfen/PyBoy?tab=readme-ov-file)

Pokemon项目：[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments/blob/master/v2/red_gym_env_v2.py)