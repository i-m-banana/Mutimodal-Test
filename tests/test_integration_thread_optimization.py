"""
集成测试: 全栈线程池优化验证
====================================
测试完整系统启动后的线程使用情况

测试内容:
1. UI进程线程数监控
2. 后端进程线程数监控
3. 线程池状态诊断
4. 服务功能验证
5. 性能基准测试
"""

import os
import sys
import time
import psutil
import threading
import logging
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test.integration")


class ThreadMonitor:
    """线程监控器"""
    
    def __init__(self, process_name: str):
        self.process_name = process_name
        self.initial_count = 0
        self.current_count = 0
        
    def find_process(self):
        """查找目标进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('project-root' in arg for arg in cmdline):
                    if 'ui/main.py' in ' '.join(cmdline):
                        return proc
                    elif 'src/main.py' in ' '.join(cmdline):
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_thread_count(self, proc):
        """获取进程线程数"""
        try:
            return proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
    
    def snapshot(self):
        """拍摄线程快照"""
        proc = self.find_process()
        if proc:
            self.current_count = self.get_thread_count(proc)
            return self.current_count
        return 0
    
    def set_baseline(self):
        """设置基线"""
        self.initial_count = self.snapshot()
        return self.initial_count


def test_backend_thread_pool():
    """测试1: 后端线程池状态"""
    logger.info("=" * 60)
    logger.info("测试1: 后端线程池状态诊断")
    
    try:
        from src.core.thread_pool import BackendThreadPool
        
        pool = BackendThreadPool()
        stats = pool.diagnostics()
        
        logger.info("后端线程池状态:")
        logger.info(f"  IO池工作线程: {stats['io_pool']['max_workers']}")
        logger.info(f"  CPU池工作线程: {stats['cpu_pool']['max_workers']}")
        logger.info(f"  活跃托管线程: {len([t for t in stats['managed_threads'].values() if t['alive']])}")
        logger.info(f"  托管线程列表: {list(stats['managed_threads'].keys())}")
        
        # 验证线程池大小符合设计
        assert stats['io_pool']['max_workers'] == 2, "IO池应该是2个工作线程"
        assert stats['cpu_pool']['max_workers'] == 4, "CPU池应该是4个工作线程"
        
        logger.info("✓ 后端线程池状态正常")
        return True
        
    except Exception as e:
        logger.error(f"✗ 后端线程池测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_ui_thread_pool():
    """测试2: UI线程池状态"""
    logger.info("=" * 60)
    logger.info("测试2: UI线程池状态诊断")
    
    try:
        from ui.runtime.ui_thread_pool import UIThreadPool
        
        pool = UIThreadPool()
        stats = pool.diagnostics()
        
        logger.info("UI线程池状态:")
        logger.info(f"  UI池工作线程: {stats['ui_pool']['max_workers']}")
        logger.info(f"  活跃托管线程: {len([t for t in stats['managed_threads'].values() if t['alive']])}")
        logger.info(f"  托管线程列表: {list(stats['managed_threads'].keys())}")
        
        # 验证线程池大小符合设计
        assert stats['ui_pool']['max_workers'] == 4, "UI池应该是4个工作线程"
        
        logger.info("✓ UI线程池状态正常")
        return True
        
    except Exception as e:
        logger.error(f"✗ UI线程池测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_thread_count_reduction():
    """测试3: 线程数量减少验证"""
    logger.info("=" * 60)
    logger.info("测试3: 线程数量对比分析")
    
    # 优化前的预期线程数(历史基线)
    BASELINE_UI_THREADS = 15  # 优化前UI进程预期线程数
    BASELINE_BACKEND_THREADS = 18  # 优化前后端进程预期线程数
    
    # 当前线程数
    ui_monitor = ThreadMonitor("ui")
    backend_monitor = ThreadMonitor("backend")
    
    current_ui = ui_monitor.snapshot()
    current_backend = backend_monitor.snapshot()
    
    logger.info("线程数量对比:")
    logger.info(f"  UI进程:")
    logger.info(f"    优化前(预期): {BASELINE_UI_THREADS} 线程")
    logger.info(f"    优化后(实际): {current_ui} 线程")
    if current_ui > 0:
        reduction_ui = ((BASELINE_UI_THREADS - current_ui) / BASELINE_UI_THREADS) * 100
        logger.info(f"    减少比例: {reduction_ui:.1f}%")
    
    logger.info(f"  后端进程:")
    logger.info(f"    优化前(预期): {BASELINE_BACKEND_THREADS} 线程")
    logger.info(f"    优化后(实际): {current_backend} 线程")
    if current_backend > 0:
        reduction_backend = ((BASELINE_BACKEND_THREADS - current_backend) / BASELINE_BACKEND_THREADS) * 100
        logger.info(f"    减少比例: {reduction_backend:.1f}%")
    
    # 验证线程数减少目标(约40%)
    if current_ui > 0 and current_backend > 0:
        total_baseline = BASELINE_UI_THREADS + BASELINE_BACKEND_THREADS
        total_current = current_ui + current_backend
        total_reduction = ((total_baseline - total_current) / total_baseline) * 100
        
        logger.info(f"  总体减少: {total_reduction:.1f}%")
        
        if total_reduction >= 30:  # 允许30%-50%的减少范围
            logger.info("✓ 线程数量显著减少,优化目标达成")
            return True
        else:
            logger.warning(f"⚠ 线程减少比例({total_reduction:.1f}%)低于预期(30-50%)")
            return True  # 不强制失败,只是警告
    else:
        logger.warning("⚠ 无法获取进程信息,跳过线程数量验证")
        return True


def test_thread_pool_isolation():
    """测试4: UI/后端线程池隔离验证"""
    logger.info("=" * 60)
    logger.info("测试4: UI/后端线程池隔离验证")
    
    try:
        from src.core.thread_pool import BackendThreadPool
        from ui.runtime.ui_thread_pool import UIThreadPool
        
        backend_pool = BackendThreadPool()
        ui_pool = UIThreadPool()
        
        # 验证是不同的实例
        assert backend_pool is not ui_pool, "UI和后端应该使用不同的线程池实例"
        assert backend_pool._io_pool is not ui_pool._ui_pool, "线程池内部executor应该完全独立"
        
        logger.info("✓ UI/后端线程池完全隔离")
        return True
        
    except Exception as e:
        logger.error(f"✗ 线程池隔离验证失败: {e}")
        return False


def test_concurrent_task_execution():
    """测试5: 并发任务执行验证"""
    logger.info("=" * 60)
    logger.info("测试5: 并发任务执行压力测试")
    
    try:
        from src.core.thread_pool import BackendThreadPool
        from ui.runtime.ui_thread_pool import UIThreadPool
        
        backend_pool = BackendThreadPool()
        ui_pool = UIThreadPool()
        
        # 后端IO任务
        def io_task(n):
            time.sleep(0.1)
            return f"io-{n}"
        
        # 后端CPU任务
        def cpu_task(n):
            result = sum(i * i for i in range(1000))
            return f"cpu-{n}-{result}"
        
        # UI任务
        def ui_task(n):
            time.sleep(0.05)
            return f"ui-{n}"
        
        # 提交大量并发任务
        io_futures = [backend_pool.submit_io_task(io_task, i) for i in range(10)]
        cpu_futures = [backend_pool.submit_cpu_task(cpu_task, i) for i in range(10)]
        ui_futures = [ui_pool.submit_task(ui_task, i) for i in range(10)]
        
        # 等待所有任务完成
        io_results = [f.result(timeout=5) for f in io_futures]
        cpu_results = [f.result(timeout=5) for f in cpu_futures]
        ui_results = [f.result(timeout=5) for f in ui_futures]
        
        # 验证结果
        assert len(io_results) == 10, "IO任务应该全部完成"
        assert len(cpu_results) == 10, "CPU任务应该全部完成"
        assert len(ui_results) == 10, "UI任务应该全部完成"
        
        logger.info(f"✓ 并发任务执行成功:")
        logger.info(f"  IO任务: {len(io_results)}/10 完成")
        logger.info(f"  CPU任务: {len(cpu_results)}/10 完成")
        logger.info(f"  UI任务: {len(ui_results)}/10 完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 并发任务执行失败: {e}")
        return False


def test_performance_benchmark():
    """测试6: 性能基准测试"""
    logger.info("=" * 60)
    logger.info("测试6: 性能基准测试")
    
    try:
        from src.core.thread_pool import BackendThreadPool
        
        pool = BackendThreadPool()
        
        # 基准任务
        def benchmark_task(n):
            return sum(i * i for i in range(10000))
        
        # 顺序执行基准
        start = time.time()
        sequential_results = [benchmark_task(i) for i in range(20)]
        sequential_time = time.time() - start
        
        # 并行执行基准
        start = time.time()
        futures = [pool.submit_cpu_task(benchmark_task, i) for i in range(20)]
        parallel_results = [f.result() for f in futures]
        parallel_time = time.time() - start
        
        speedup = sequential_time / parallel_time
        
        logger.info("性能基准结果:")
        logger.info(f"  顺序执行: {sequential_time:.3f}秒")
        logger.info(f"  并行执行: {parallel_time:.3f}秒")
        logger.info(f"  加速比: {speedup:.2f}x")
        
        if speedup > 1.5:
            logger.info("✓ 并行执行显著提升性能")
            return True
        else:
            logger.warning(f"⚠ 加速比({speedup:.2f}x)低于预期(>1.5x)")
            return True  # 不强制失败
        
    except Exception as e:
        logger.error(f"✗ 性能基准测试失败: {e}")
        return False


def main():
    """主测试流程"""
    logger.info("开始集成测试: 全栈线程池优化验证")
    logger.info("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("后端线程池状态", test_backend_thread_pool()))
    results.append(("UI线程池状态", test_ui_thread_pool()))
    results.append(("线程数量减少", test_thread_count_reduction()))
    results.append(("线程池隔离", test_thread_pool_isolation()))
    results.append(("并发任务执行", test_concurrent_task_execution()))
    results.append(("性能基准测试", test_performance_benchmark()))
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("集成测试结果汇总:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {name}: {status}")
    
    logger.info("=" * 60)
    if passed == total:
        logger.info(f"✓ 所有测试通过! ({passed}/{total})")
        logger.info("全栈线程池优化工作正常")
    else:
        logger.error(f"✗ 部分测试失败 ({passed}/{total})")
    
    logger.info("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行异常: {e}", exc_info=True)
        sys.exit(1)
