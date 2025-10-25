# Envs
之前采取的环境封装策略比较粗糙，所有房间的reward function都扭在一个文件中，导致越拖越长，文件也快成尸山了

现在采取划分抽象类的方式，封装一个BaseEnv作为基类，其中包含一些通用函数和Gym的API接口，而其他房间则继承BaseEnv并自己实现对应房间的任务函数


# Appendix 
## BaseEnv
### 通用函数

**preprocess_for_rl**

负责将图像信号进行压缩并输出单通道信号(目前由于输入的是game area暂时不需要这个)
```python
def preprocess_for_rl(self):
    game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
    game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2, 1)).astype(np.uint8)
    return game_pixels_render
```

**_get_pos**

输出 Link 当前在房间中的位置
```python
def _get_pos(self) -> Tuple[int, int]:
    sprite = self.pyboy.get_sprite(2)
    return sprite.x, sprite.y
```

**is_dead**

检测 Link 是否死亡💀
```python
def is_dead(self) -> bool:
    return self.read_m(ADDR_CUR_HEALTH) == 0
```

**is_hurt**

检测 Link 是否受伤，如果受伤返回收到的伤害大小
```python
def is_hurt(self) -> int:
    if not isinstance(self.cur_health, (int, float)) or not isinstance(self.pre_health, (int, float)):
        return 0
    if self.cur_health < self.pre_health:
        return self.cur_health - self.pre_health
    return 0
```

**outside_counter_tick**

如果一直在房间外游荡则强制重启（返回true）
```python
def outside_counter_tick(self, max_out: int = 100) -> bool:
    if self.cur_room != self.goal_room:
        self.out_side += 1
    else:
        self.out_side = 0
    if self.out_side >= max_out:
        self.out_side = 0
        return True
    return False
```

**tile_explore_bonus**

```python
def tile_explore_bonus(self) -> bool:
    if self.cur_room == self.goal_room:
        tile_x, tile_y = self._get_tile()
        key = (int(self.cur_room), tile_x, tile_y)
        if key not in self.visited_tiles:
            self.visited_tiles.add(key)
            return True
    return False
```