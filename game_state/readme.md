# Game State
这个目录下汇总了各个房间所需要的游戏状态（State）

## Room 51
![task1](/img/room51.png)
为了方便训练，将51号房间划分成不同的任务
**Task one**:无怪物模拟

对应文件为 `Room51_task1.state` 和 `env51_01.py`

任务一计划在没有粘液怪物和乌龟怪物的前提下，让agent学习到绕过中央 凹 型陷阱并才下按钮的能力，踩下按钮之后会在右上方出现宝箱，靠近包厢并按下 A button即可获得钥匙（key）

## Room 58
**Task one**:无怪物模拟

对应文件为 `Room58_task1.state` 和 `env58_01.py`

任务目标是在没有两只甲虫怪物的叨扰下，前往钥匙所在位置拿到key（30，45）