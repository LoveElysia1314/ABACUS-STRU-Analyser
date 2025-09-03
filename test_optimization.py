#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化效果的简单脚本
比较传统模式与流式模式的性能差异
"""

import os
import time
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from src.io.path_manager import PathManager
from src.utils.common import FileUtils

def test_discovery_performance(search_paths, use_streaming=False, use_cache=True):
    """测试发现性能"""
    print(f"\n{'='*60}")
    mode = "流式模式" if use_streaming else "传统模式"
    cache_info = "启用缓存" if use_cache else "禁用缓存"
    print(f"测试 {mode} ({cache_info})")
    print(f"{'='*60}")
    
    # 发现体系
    start_time = time.time()
    all_mol_systems = {}
    for search_path in search_paths:
        print(f"搜索路径: {search_path}")
        mol_systems = FileUtils.find_abacus_systems(search_path)
        for mol_key, system_paths in mol_systems.items():
            if mol_key in all_mol_systems:
                all_mol_systems[mol_key].extend(system_paths)
            else:
                all_mol_systems[mol_key] = system_paths
    
    discovery_time = time.time() - start_time
    total_systems = sum(len(s) for s in all_mol_systems.values())
    print(f"体系发现完成: {len(all_mol_systems)} 个分子, {total_systems} 个体系, 耗时 {discovery_time:.2f}s")
    
    if total_systems == 0:
        print("未找到任何体系，测试结束")
        return
    
    # 测试加载性能
    path_manager = PathManager("test_output")
    
    if use_streaming:
        print("\n开始流式加载测试...")
        batch_count = 0
        first_batch_time = None
        start_load = time.time()
        
        for batch_targets, is_final, progress_info in path_manager.discover_targets_streaming(
            all_mol_systems, batch_size=50, use_cache=use_cache
        ):
            batch_count += 1
            if first_batch_time is None:
                first_batch_time = time.time() - start_load
                print(f"首批 {len(batch_targets)} 个体系加载完成: {first_batch_time:.2f}s")
            
            if batch_count % 5 == 0 or is_final:
                print(f"批次 {batch_count}: 累计 {progress_info['completed']}/{progress_info['total']}, "
                      f"耗时 {progress_info['elapsed']:.2f}s, "
                      f"缓存命中 {progress_info['cache_hits']}")
            
            if is_final:
                break
        
        total_load_time = time.time() - start_load
        print(f"流式加载完成: 总耗时 {total_load_time:.2f}s, 首批耗时 {first_batch_time:.2f}s")
        
    else:
        print("\n开始传统加载测试...")
        start_load = time.time()
        path_manager.load_from_discovery(search_paths, use_cache=use_cache)
        total_load_time = time.time() - start_load
        print(f"传统加载完成: {len(path_manager.targets)} 个目标, 耗时 {total_load_time:.2f}s")
    
    total_time = discovery_time + total_load_time
    print(f"\n总结:")
    print(f"  发现耗时: {discovery_time:.2f}s")
    print(f"  加载耗时: {total_load_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    if use_streaming and first_batch_time:
        print(f"  首批延迟: {first_batch_time:.2f}s (相比总耗时减少 {(total_time-first_batch_time)/total_time*100:.1f}%)")
    
    return {
        'mode': mode,
        'use_cache': use_cache,
        'discovery_time': discovery_time,
        'load_time': total_load_time,
        'total_time': total_time,
        'first_batch_time': first_batch_time,
        'total_systems': total_systems
    }

def main():
    parser = argparse.ArgumentParser(description='测试发现性能优化效果')
    parser.add_argument('-s', '--search_path', nargs='*', default=None, 
                       help='搜索路径 (默认当前目录的父目录)')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
    parser.add_argument('--traditional-only', action='store_true', help='仅测试传统模式')
    
    args = parser.parse_args()
    
    # 设置搜索路径
    if args.search_path:
        search_paths = args.search_path
    else:
        search_paths = [os.path.abspath(os.path.join(os.getcwd(), '..'))]
    
    print("ABACUS-STRU-Analyser 发现性能测试")
    print(f"搜索路径: {search_paths}")
    
    results = []
    use_cache = not args.no_cache
    
    # 测试传统模式
    try:
        result = test_discovery_performance(search_paths, use_streaming=False, use_cache=use_cache)
        results.append(result)
    except Exception as e:
        print(f"传统模式测试失败: {e}")
    
    # 测试流式模式
    if not args.traditional_only:
        try:
            result = test_discovery_performance(search_paths, use_streaming=True, use_cache=use_cache)
            results.append(result)
        except Exception as e:
            print(f"流式模式测试失败: {e}")
    
    # 性能对比
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("性能对比总结")
        print(f"{'='*60}")
        
        traditional = results[0]
        streaming = results[1]
        
        print(f"体系总数: {traditional['total_systems']}")
        print(f"传统模式总耗时: {traditional['total_time']:.2f}s")
        print(f"流式模式总耗时: {streaming['total_time']:.2f}s")
        
        if streaming['first_batch_time']:
            first_batch_speedup = traditional['total_time'] / streaming['first_batch_time']
            print(f"首批延迟减少: {first_batch_speedup:.1f}x 倍")
        
        if streaming['total_time'] < traditional['total_time']:
            total_speedup = traditional['total_time'] / streaming['total_time']
            print(f"总体性能提升: {total_speedup:.1f}x 倍")
        else:
            slowdown = streaming['total_time'] / traditional['total_time']
            print(f"总体性能影响: {slowdown:.1f}x 倍 (轻微减缓，但首批延迟大幅降低)")
    
    print(f"\n缓存目录: {os.path.join('test_output', '.cache')}")
    print("测试完成!")

if __name__ == "__main__":
    main()
