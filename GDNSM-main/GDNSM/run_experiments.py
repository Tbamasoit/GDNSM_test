import os
import time
import subprocess
import itertools
from datetime import datetime

# ================= 配置区域 =================

# 1. 基础命令 (你的入口文件)
BASE_CMD = "python GDNSM-main/GDNSM/main.py --dataset baby"

# 2. 定义你要搜索的参数空间 (Grid Search)
# 脚本会自动生成这些列表的【笛卡尔积】组合
param_grid = {
    'reverse': [1, 2],          # UFN 挖掘阈值
    'lbd': [0.1, 0.5, 1.0],     # UFN Loss 权重
    'ufn_warmup': [0, 3, 5],       # UFN 介入时机
    'decay': [0.999],           # Teacher EMA 衰减
    # 'smoothing_S':[10,9],
    
    # 你还可以加 GDNSM 的参数
    # 'sched_S': [10, 20, 30], 
}

# 3. 日志文件名
LOG_FILE = "experiment_log.txt"

# ===========================================

def get_combinations(grid):
    """将参数字典转换为参数组合列表"""
    keys = grid.keys()
    values = grid.values()
    combinations = []
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))
    return combinations


def run():
    # 生成所有实验组合
    experiments = get_combinations(param_grid)
    total_exp = len(experiments)
    
    print(f"🚀 准备开始自动化实验，共计 {total_exp} 组任务...")
    print(f"📝 日志将记录在: {LOG_FILE}\n")

    # 记录总开始时间
    global_start = time.time()

    for i, params in enumerate(experiments):
        exp_idx = i + 1
        
        # 1. 构建命令行参数
        cmd_args = []
        for key, value in params.items():
            cmd_args.append(f"--{key} {value}")
        
        full_cmd = f"{BASE_CMD} {' '.join(cmd_args)}"
        
        # 2. 打印当前任务信息
        start_time = datetime.now()
        print("="*60)
        print(f"▶️  正在执行第 [{exp_idx}/{total_exp}] 组实验")
        print(f"⚙️  参数: {params}")
        print(f"💻  命令: {full_cmd}")
        print(f"⏰  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # 3. 执行命令
        # try-except 确保即使某个实验报错，脚本也能继续跑下一个
        exp_start_time = time.time()
        status = "SUCCESS"
        try:
            # shell=True 允许执行完整的字符串命令
            # check=True 会在命令返回非0状态码时抛出异常
            subprocess.run(full_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            status = "FAILED"
            print(f"\n❌ 实验 [{exp_idx}] 失败! Error Code: {e.returncode}")
        except KeyboardInterrupt:
            print("\n🛑 用户手动终止脚本。")
            break
        
        # 4. 记录耗时
        exp_end_time = time.time()
        duration = exp_end_time - exp_start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        duration_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

        print(f"\n✅ 实验 [{exp_idx}] 结束. 状态: {status}. 耗时: {duration_str}")

        # 5. 写入日志文件
        with open(LOG_FILE, "a") as f:
            log_line = (
                f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"ID={exp_idx}/{total_exp} | "
                f"Status={status} | "
                f"Duration={duration_str} | "
                f"Params={params}\n"
            )
            f.write(log_line)
            
    # 总耗时
    total_duration = time.time() - global_start
    print("\n" + "="*60)
    print(f"🎉 所有任务执行完毕！总耗时: {total_duration/3600:.2f} 小时")
    print("="*60)

if __name__ == "__main__":
    run()









