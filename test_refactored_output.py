#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试重构后脚本的输出格式
"""

import subprocess
import sys
import os

def run_test():
    """运行测试"""
    print("=" * 60)
    print("测试重构后的主流程脚本输出格式")
    print("=" * 60)
    
    # 准备测试命令
    script_path = "main_abacus_analyser_refactored.py"
    test_args = [
        sys.executable, script_path,
        "--dry_run_reuse",  # 使用dry run模式避免实际计算
        "-r", "0.1",
        "-w", "2",
        "--scheduler", "process"
    ]
    
    print(f"执行命令: {' '.join(test_args)}")
    print("-" * 60)
    
    try:
        # 运行脚本
        result = subprocess.run(
            test_args,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\n返回码: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("测试超时")
    except Exception as e:
        print(f"测试出错: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    run_test()
