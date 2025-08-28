
ABACUS STRU 轨迹分析器（优化采样版）
====================================

功能：
-----
1. 批量分析ABACUS分子动力学模拟的STRU文件
2. 计算构象多样性核心指标
3. 基于幂平均距离（Power Mean）优化采样策略（支持p=1算数平均，p=0几何平均，p=-1调和平均，默认p=0.5）
4. 输出单帧指标和系统汇总

核心指标：
--------
1. ConfVol: 构象空间体积（核心多样性指标）
2. nRMSF: 标准化距离均方根波动
3. MCV: 平均变异系数
4. nLdRMS: 每帧到平均结构的距离RMS


采样策略：
---------
使用贪心算法最大化采样点间的幂平均距离（Power Mean），支持不同p值：
    p=1   —— 算数平均距离（Arithmetic Mean）
    p=0   —— 几何平均距离（Geometric Mean）
    p=-1  —— 调和平均距离（Harmonic Mean）
    其它p —— 幂平均距离
默认p=0.5。
通过 --power_p 参数设置。

输入结构：
---------
当前目录下包含多个体系文件夹：
    struct_mol_<ID>_conf_<N>_T<T>K/
    └── OUT.ABACUS/
        └── STRU/
            ├── STRU_MD_0
            ├── STRU_MD_1
            └── ...

输出结构：
--------
analysis_results/
├── struct_mol_<ID>/
│   └── metrics_per_frame_<system>.csv
├── system_summary.csv
└── analysis.log  # 新增日志文件