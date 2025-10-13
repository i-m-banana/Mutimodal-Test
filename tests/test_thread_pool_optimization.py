"""验证后端线程池优化的测试脚本."""

import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.thread_pool import get_thread_pool
from src.core.event_bus import EventBus
from src.services.multimodal_service import MultimodalService
from src.services.av_service import AVService
from src.services.bp_service import BloodPressureService
from src.services.tts_service import TTSService
from src.services.ui_command_router import UICommandRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("test.thread_pool")


def test_thread_pool_singleton():
    """测试线程池单例模式."""
    logger.info("=" * 60)
    logger.info("测试1: 线程池单例模式")
    
    pool1 = get_thread_pool()
    pool2 = get_thread_pool()
    
    assert pool1 is pool2, "线程池应该是单例"
    logger.info("✓ 线程池单例验证通过")
    
    # 获取诊断信息
    diag = pool1.diagnostics()
    logger.info("线程池状态:")
    logger.info(f"  IO线程池: max_workers={diag['io_pool']['max_workers']}")
    logger.info(f"  CPU线程池: max_workers={diag['cpu_pool']['max_workers']}")
    logger.info(f"  托管线程数: {len(diag['managed_threads'])}")


def test_managed_threads():
    """测试托管线程注册和注销."""
    logger.info("=" * 60)
    logger.info("测试2: 托管线程管理")
    
    pool = get_thread_pool()
    
    # 注册测试线程
    stop_event = [False]
    
    def test_worker():
        logger.info("测试工作线程启动")
        while not stop_event[0]:
            time.sleep(0.1)
        logger.info("测试工作线程停止")
    
    thread = pool.register_managed_thread("test-worker", test_worker, daemon=True)
    thread.start()
    
    # 验证线程运行
    time.sleep(0.3)
    assert thread.is_alive(), "托管线程应该正在运行"
    logger.info("✓ 托管线程注册成功")
    
    # 停止并注销
    stop_event[0] = True
    pool.unregister_managed_thread("test-worker", timeout=2.0)
    
    time.sleep(0.2)
    assert not thread.is_alive(), "托管线程应该已停止"
    logger.info("✓ 托管线程注销成功")


def test_io_cpu_pools():
    """测试IO和CPU线程池."""
    logger.info("=" * 60)
    logger.info("测试3: IO/CPU线程池")
    
    pool = get_thread_pool()
    
    results = {"io": [], "cpu": []}
    
    def io_task(name):
        logger.info(f"IO任务执行: {name}")
        time.sleep(0.1)
        results["io"].append(name)
        return f"io-{name}"
    
    def cpu_task(name):
        logger.info(f"CPU任务执行: {name}")
        time.sleep(0.1)
        results["cpu"].append(name)
        return f"cpu-{name}"
    
    # 提交IO任务
    io_futures = [pool.submit_io_task(io_task, f"task-{i}") for i in range(3)]
    
    # 提交CPU任务
    cpu_futures = [pool.submit_cpu_task(cpu_task, f"task-{i}") for i in range(3)]
    
    # 等待完成
    for f in io_futures:
        result = f.result(timeout=2.0)
        logger.info(f"  IO结果: {result}")
    
    for f in cpu_futures:
        result = f.result(timeout=2.0)
        logger.info(f"  CPU结果: {result}")
    
    assert len(results["io"]) == 3, "应该完成3个IO任务"
    assert len(results["cpu"]) == 3, "应该完成3个CPU任务"
    logger.info("✓ IO/CPU线程池验证通过")


def test_services_integration():
    """测试服务集成."""
    logger.info("=" * 60)
    logger.info("测试4: 服务集成验证")
    
    bus = EventBus()
    pool = get_thread_pool()
    
    # 初始化服务
    logger.info("初始化服务...")
    multimodal = MultimodalService(bus=bus, logger=logger.getChild("multimodal"))
    av = AVService(bus, logger=logger.getChild("av"))
    bp = BloodPressureService(logger=logger.getChild("bp"))
    tts = TTSService(logger=logger.getChild("tts"))
    
    # 检查线程状态
    time.sleep(0.5)
    diag = pool.diagnostics()
    logger.info(f"服务初始化后托管线程数: {len(diag['managed_threads'])}")
    
    for name, info in diag['managed_threads'].items():
        logger.info(f"  {name}: alive={info['alive']}, daemon={info['daemon']}")
    
    # TTS应该已启动worker线程
    assert "tts-worker" in diag['managed_threads'], "TTS worker应该已注册"
    assert diag['managed_threads']['tts-worker']['alive'], "TTS worker应该正在运行"
    logger.info("✓ 服务集成验证通过")
    
    # 清理
    tts.shutdown()
    av.shutdown()
    bp.stop()


def test_command_router():
    """测试UI命令路由器."""
    logger.info("=" * 60)
    logger.info("测试5: UI命令路由器")
    
    bus = EventBus()
    router = UICommandRouter(bus, logger=logger.getChild("router"))
    
    router.start()
    logger.info("命令路由器已启动")
    
    time.sleep(0.3)
    pool = get_thread_pool()
    diag = pool.diagnostics()
    
    logger.info(f"路由器启动后托管线程数: {len(diag['managed_threads'])}")
    logger.info("✓ UI命令路由器验证通过")
    
    router.stop()
    logger.info("命令路由器已停止")


def main():
    """运行所有测试."""
    logger.info("开始后端线程池优化验证")
    logger.info("=" * 60)
    
    try:
        test_thread_pool_singleton()
        test_managed_threads()
        test_io_cpu_pools()
        test_services_integration()
        test_command_router()
        
        logger.info("=" * 60)
        logger.info("✓ 所有测试通过!")
        logger.info("后端线程池优化工作正常")
        
        # 最终诊断
        pool = get_thread_pool()
        final_diag = pool.diagnostics()
        logger.info("=" * 60)
        logger.info("最终线程池状态:")
        logger.info(f"  IO线程池: max_workers={final_diag['io_pool']['max_workers']}")
        logger.info(f"  CPU线程池: max_workers={final_diag['cpu_pool']['max_workers']}")
        logger.info(f"  活跃托管线程: {sum(1 for t in final_diag['managed_threads'].values() if t['alive'])}")
        logger.info(f"  总托管线程: {len(final_diag['managed_threads'])}")
        
    except Exception as exc:
        logger.error(f"✗ 测试失败: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
