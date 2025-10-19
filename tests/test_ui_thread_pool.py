"""UI层线程池优化验证测试."""

import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.runtime.ui_thread_pool import get_ui_thread_pool
from ui.services.backend_client import BackendClient

# 可选导入 (避免依赖问题)
try:
    from ui.services.av_service import _RemoteAVProxy
    HAS_AV_SERVICE = True
except ImportError:
    HAS_AV_SERVICE = False
    _RemoteAVProxy = None  # type: ignore

from ui.services.backend_launcher import BackendLauncher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("test.ui_thread_pool")


def test_ui_thread_pool_singleton():
    """测试UI线程池单例模式."""
    logger.info("=" * 60)
    logger.info("测试1: UI线程池单例模式")
    
    pool1 = get_ui_thread_pool()
    pool2 = get_ui_thread_pool()
    
    assert pool1 is pool2, "UI线程池应该是单例"
    logger.info("✓ UI线程池单例验证通过")
    
    # 获取诊断信息
    diag = pool1.diagnostics()
    logger.info("UI线程池状态:")
    logger.info(f"  UI任务线程池: max_workers={diag['ui_pool']['max_workers']}")
    logger.info(f"  托管线程数: {len(diag['managed_threads'])}")


def test_ui_task_submission():
    """测试UI任务提交."""
    logger.info("=" * 60)
    logger.info("测试2: UI任务提交")
    
    pool = get_ui_thread_pool()
    
    results = []
    
    def simple_task(name, value):
        logger.info(f"执行UI任务: {name} = {value}")
        time.sleep(0.1)
        results.append((name, value * 2))
        return value * 2
    
    # 提交多个任务
    futures = [
        pool.submit_task(simple_task, f"task-{i}", i)
        for i in range(5)
    ]
    
    # 等待所有任务完成
    for i, future in enumerate(futures):
        result = future.result(timeout=2.0)
        logger.info(f"  任务{i}结果: {result}")
        assert result == i * 2, f"任务{i}结果错误"
    
    assert len(results) == 5, "应该完成5个任务"
    logger.info("✓ UI任务提交验证通过")


def test_ui_managed_thread():
    """测试UI托管线程."""
    logger.info("=" * 60)
    logger.info("测试3: UI托管线程管理")
    
    pool = get_ui_thread_pool()
    
    # 测试数据
    counter = {"value": 0}
    stop_flag = [False]
    
    def worker_loop():
        logger.info("UI测试工作线程启动")
        while not stop_flag[0]:
            counter["value"] += 1
            time.sleep(0.1)
        logger.info("UI测试工作线程停止")
    
    # 注册托管线程
    thread = pool.register_managed_thread(
        "ui-test-worker",
        worker_loop,
        daemon=True
    )
    thread.start()
    
    # 验证线程运行
    time.sleep(0.5)
    assert thread.is_alive(), "托管线程应该正在运行"
    assert counter["value"] > 0, "工作线程应该在执行"
    logger.info(f"✓ UI托管线程注册成功 (执行次数: {counter['value']})")
    
    # 停止并注销
    stop_flag[0] = True
    pool.unregister_managed_thread("ui-test-worker", timeout=2.0)
    
    time.sleep(0.2)
    assert not thread.is_alive(), "托管线程应该已停止"
    logger.info("✓ UI托管线程注销成功")


def test_backend_client_integration():
    """测试后端客户端集成(不实际连接)."""
    logger.info("=" * 60)
    logger.info("测试4: 后端客户端集成")
    
    pool = get_ui_thread_pool()
    
    # 创建客户端(但不启动连接)
    client = BackendClient(url="ws://127.0.0.1:9999")
    
    # 检查线程池使用
    assert hasattr(client, '_thread_pool'), "客户端应该有线程池引用"
    assert client._thread_pool is pool, "客户端应该使用UI线程池"
    
    logger.info("✓ 后端客户端集成验证通过")


def test_av_proxy_integration():
    """测试AV代理集成."""
    logger.info("=" * 60)
    logger.info("测试5: AV代理集成")
    
    if not HAS_AV_SERVICE:
        logger.warning("⚠ AV服务不可用(缺少cv2),跳过测试")
        return
    
    pool = get_ui_thread_pool()
    
    # 创建AV代理
    av_proxy = _RemoteAVProxy()
    
    # 检查线程池使用
    assert hasattr(av_proxy, '_thread_pool'), "AV代理应该有线程池引用"
    assert av_proxy._thread_pool is pool, "AV代理应该使用UI线程池"
    
    logger.info("✓ AV代理集成验证通过")


def test_backend_launcher_integration():
    """测试后端启动器集成(不实际启动后端)."""
    logger.info("=" * 60)
    logger.info("测试6: 后端启动器集成")
    
    pool = get_ui_thread_pool()
    
    # 创建启动器
    launcher = BackendLauncher()
    
    # 检查线程池使用
    assert hasattr(launcher, '_thread_pool'), "启动器应该有线程池引用"
    assert launcher._thread_pool is pool, "启动器应该使用UI线程池"
    
    logger.info("✓ 后端启动器集成验证通过")


def test_thread_separation():
    """测试UI线程池与后端线程池分离."""
    logger.info("=" * 60)
    logger.info("测试7: UI/后端线程池分离")
    
    from src.core.thread_pool import get_thread_pool
    
    ui_pool = get_ui_thread_pool()
    backend_pool = get_thread_pool()
    
    assert ui_pool is not backend_pool, "UI和后端线程池应该是不同的实例"
    logger.info("✓ UI/后端线程池分离验证通过")
    
    # 显示两个池的状态
    ui_diag = ui_pool.diagnostics()
    backend_diag = backend_pool.diagnostics()
    
    logger.info("线程池对比:")
    logger.info(f"  UI线程池: {ui_diag['ui_pool']['max_workers']}个工作线程")
    logger.info(f"  后端IO池: {backend_diag['io_pool']['max_workers']}个工作线程")
    logger.info(f"  后端CPU池: {backend_diag['cpu_pool']['max_workers']}个工作线程")


def main():
    """运行所有测试."""
    logger.info("开始UI层线程池优化验证")
    logger.info("=" * 60)
    
    try:
        test_ui_thread_pool_singleton()
        test_ui_task_submission()
        test_ui_managed_thread()
        test_backend_client_integration()
        test_av_proxy_integration()
        test_backend_launcher_integration()
        test_thread_separation()
        
        logger.info("=" * 60)
        logger.info("✓ 所有测试通过!")
        logger.info("UI层线程池优化工作正常")
        
        # 最终诊断
        pool = get_ui_thread_pool()
        final_diag = pool.diagnostics()
        logger.info("=" * 60)
        logger.info("最终UI线程池状态:")
        logger.info(f"  UI任务线程池: max_workers={final_diag['ui_pool']['max_workers']}")
        logger.info(f"  活跃托管线程: {sum(1 for t in final_diag['managed_threads'].values() if t['alive'])}")
        logger.info(f"  总托管线程: {len(final_diag['managed_threads'])}")
        
    except Exception as exc:
        logger.error(f"✗ 测试失败: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
