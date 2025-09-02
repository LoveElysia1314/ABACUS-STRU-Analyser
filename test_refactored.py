#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：验证重构后的ABACUS分析器
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_sampling_only_mode():
    """测试仅采样模式"""
    print("测试仅采样模式...")
    
    # 模拟命令行参数
    import argparse
    sys.argv = [
        'test_script.py',
        '--sampling_only',
        '--sample_ratio', '0.1',
        '--workers', '1',
        '--dry_run_reuse'  # 仅测试流程，不实际处理
    ]
    
    try:
        from main_abacus_analyser_refactored import MainApp
        app = MainApp()
        app.run()
        print("✓ 仅采样模式测试通过")
    except Exception as e:
        print(f"✗ 仅采样模式测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_full_analysis_mode():
    """测试完整分析模式"""
    print("测试完整分析模式...")
    
    # 模拟命令行参数
    import argparse
    sys.argv = [
        'test_script.py',
        '--sample_ratio', '0.05',
        '--workers', '1',
        '--dry_run_reuse'  # 仅测试流程，不实际处理
    ]
    
    try:
        from main_abacus_analyser_refactored import MainApp
        app = MainApp()
        app.run()
        print("✓ 完整分析模式测试通过")
    except Exception as e:
        print(f"✗ 完整分析模式测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_config_creation():
    """测试配置对象创建"""
    print("测试配置对象创建...")
    
    try:
        from main_abacus_analyser_refactored import AnalysisConfig, AnalysisMode
        
        # 测试仅采样配置
        config1 = AnalysisConfig(
            mode=AnalysisMode.SAMPLING_ONLY,
            sample_ratio=0.1,
            workers=2
        )
        assert config1.mode == AnalysisMode.SAMPLING_ONLY
        assert config1.sample_ratio == 0.1
        assert config1.workers == 2
        
        # 测试完整分析配置
        config2 = AnalysisConfig(
            mode=AnalysisMode.FULL_ANALYSIS,
            enable_sampling_eval=False
        )
        assert config2.mode == AnalysisMode.FULL_ANALYSIS
        assert config2.enable_sampling_eval == False
        
        print("✓ 配置对象创建测试通过")
    except Exception as e:
        print(f"✗ 配置对象创建测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_orchestrator_creation():
    """测试编排器创建"""
    print("测试编排器创建...")
    
    try:
        from main_abacus_analyser_refactored import AnalysisOrchestrator, AnalysisConfig, AnalysisMode
        
        config = AnalysisConfig(mode=AnalysisMode.SAMPLING_ONLY)
        orchestrator = AnalysisOrchestrator(config)
        
        assert orchestrator.config.mode == AnalysisMode.SAMPLING_ONLY
        assert orchestrator.logger is None  # 尚未初始化
        
        print("✓ 编排器创建测试通过")
    except Exception as e:
        print(f"✗ 编排器创建测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("ABACUS分析器重构版本测试")
    print("=" * 50)
    
    # 基础测试
    test_config_creation()
    test_orchestrator_creation()
    
    # 流程测试（需要有效的数据目录）
    # test_sampling_only_mode()
    # test_full_analysis_mode()
    
    print("\n测试完成！")
    print("注意：流程测试需要有效的ABACUS数据目录，已注释掉。")
    print("如需测试完整流程，请取消注释并提供有效的搜索路径。")
