#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试模拟：显示期望的输出格式
"""

import logging
import time

def simulate_analysis():
    """模拟分析过程的输出"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('__main__')
    
    # 模拟体系名称
    systems = [
        'struct_mol_1028_conf_0_T400K',
        'struct_mol_1028_conf_1_T400K',
        'struct_mol_1050_conf_0_T400K',
        'struct_mol_1050_conf_1_T400K',
        'struct_mol_1050_conf_2_T400K'
    ]
    
    total = len(systems)
    
    logger.info("=" * 60)
    logger.info("模拟：期望的输出格式")
    logger.info("=" * 60)
    logger.info(f"开始分析 {total} 个体系...")
    
    # 模拟实时分析过程
    for i, system_name in enumerate(systems):
        # 模拟分析时间
        time.sleep(0.5)  # 模拟分析耗时
        
        # 这是我们期望在分析过程中看到的输出
        logger.info(f"({i+1}/{total}) {system_name} 体系分析完成")
        
        # 如果是仅采样模式，显示标识
        if i == 2:  # 第3个系统模拟为仅采样模式
            logger.info(f"({i+1}/{total}) {system_name} 体系分析完成 [仅采样]")
    
    logger.info("=" * 60)
    logger.info("开始保存分析结果...")
    
    # 模拟保存过程
    for i in range(total):
        if (i + 1) % 2 == 0 or (i + 1) == total:
            logger.info(f"已保存 {i + 1}/{total} 个体系的结果")
    
    logger.info("所有分析结果和DeepMD数据保存完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    simulate_analysis()
