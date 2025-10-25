#!/usr/bin/env python3
"""
数据库初始化脚本
根据 db_service.py 中的后端要求重新创建数据库表结构
"""

import pymysql
import os

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('UI_DB_HOST', 'localhost'),
    'user': os.getenv('UI_DB_USER', 'root'),
    'password': os.getenv('UI_DB_PASSWORD', '123456'),
    'charset': 'utf8mb4',
}

# 数据库名称
DATABASE_NAME = 'tired'

# 表结构SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `test` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '记录ID',
    `name` VARCHAR(100) NOT NULL COMMENT '用户名',
    `datetime` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '测试时间',
    
    -- 舒尔特方格测试结果
    `score` INT DEFAULT NULL COMMENT '舒尔特综合得分',
    `accuracy` FLOAT DEFAULT NULL COMMENT '舒尔特准确率(%)',
    `elapsed` FLOAT DEFAULT NULL COMMENT '舒尔特用时(秒)',
    
    -- 多模态数据分析结果
    `fatigue_score` FLOAT DEFAULT NULL COMMENT '疲劳检测分数',
    `brain_load_score` FLOAT DEFAULT NULL COMMENT '脑负荷分数',
    `emotion_score` FLOAT DEFAULT NULL COMMENT '情绪分数',
    
    -- 音视频路径（JSON格式存储）
    `audio` TEXT DEFAULT NULL COMMENT '音频文件路径列表(JSON)',
    `video` TEXT DEFAULT NULL COMMENT '视频文件路径列表(JSON)',
    `record` TEXT DEFAULT NULL COMMENT '文本记录列表(JSON)',
    
    -- 多模态传感器数据路径
    `rgb` VARCHAR(500) DEFAULT NULL COMMENT 'RGB视频路径',
    `depth` VARCHAR(500) DEFAULT NULL COMMENT '深度视频路径',
    `tobii` VARCHAR(500) DEFAULT NULL COMMENT '眼动数据路径',
    
    -- 生理数据
    `blood` VARCHAR(50) DEFAULT NULL COMMENT '血压数据(格式: 收缩压/舒张压/脉搏)',
    `eeg1` VARCHAR(500) DEFAULT NULL COMMENT 'EEG通道1数据路径',
    `eeg2` VARCHAR(500) DEFAULT NULL COMMENT 'EEG通道2数据路径',
    
    -- 时间戳数据
    `ptime` TEXT DEFAULT NULL COMMENT '各阶段时间戳(JSON)',
    
    -- 索引
    INDEX `idx_name` (`name`),
    INDEX `idx_datetime` (`datetime`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='多模态疲劳检测测试记录表';
"""


def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("🗄️  开始初始化数据库...")
    print("=" * 60)
    
    try:
        # 1. 连接到MySQL服务器（不指定数据库）
        print(f"\n📡 正在连接到 MySQL 服务器 {DB_CONFIG['host']}...")
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset']
        )
        cursor = conn.cursor()
        print("✅ MySQL 服务器连接成功")
        
        # 2. 创建数据库（如果不存在）
        print(f"\n📂 检查数据库 '{DATABASE_NAME}'...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✅ 数据库 '{DATABASE_NAME}' 已就绪")
        
        # 3. 切换到目标数据库
        cursor.execute(f"USE `{DATABASE_NAME}`")
        print(f"✅ 已切换到数据库 '{DATABASE_NAME}'")
        
        # 4. 检查旧表是否存在
        cursor.execute("SHOW TABLES LIKE 'test'")
        old_table_exists = cursor.fetchone() is not None
        
        if old_table_exists:
            print("\n⚠️  检测到旧的 'test' 表")
            
            # 检查是否有数据
            cursor.execute("SELECT COUNT(*) FROM `test`")
            record_count = cursor.fetchone()[0]
            print(f"   当前表中有 {record_count} 条记录")
            
            if record_count > 0:
                response = input("\n⚠️  是否要删除旧表并重新创建？这将丢失所有数据！(yes/no): ")
                if response.lower() != 'yes':
                    print("❌ 操作已取消")
                    return
            
            # 删除旧表
            print("\n🗑️  正在删除旧表...")
            cursor.execute("DROP TABLE IF EXISTS `test`")
            print("✅ 旧表已删除")
        
        # 5. 创建新表
        print("\n🔨 正在创建新表 'test'...")
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("✅ 新表创建成功")
        
        # 6. 显示表结构
        print("\n📋 表结构信息:")
        print("-" * 60)
        cursor.execute("DESCRIBE `test`")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[0]:20s} {col[1]:20s} {'NOT NULL' if col[2] == 'NO' else 'NULL':10s} {col[3] or ''}")
        
        # 7. 显示索引
        print("\n📊 索引信息:")
        print("-" * 60)
        cursor.execute("SHOW INDEX FROM `test`")
        indexes = cursor.fetchall()
        for idx in indexes:
            print(f"  {idx[2]:20s} -> {idx[4]}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ 数据库初始化完成！")
        print("=" * 60)
        print(f"\n数据库配置:")
        print(f"  - Host: {DB_CONFIG['host']}")
        print(f"  - Database: {DATABASE_NAME}")
        print(f"  - Table: test")
        print(f"  - Character Set: utf8mb4")
        print("\n后端字段映射:")
        print("  ✅ fatigue_score (疲劳检测分数)")
        print("  ✅ brain_load_score (脑负荷分数)")
        print("  ✅ emotion_score (情绪分数)")
        print("  ✅ accuracy (舒尔特准确率)")
        print("  ✅ score (舒尔特综合得分)")
        print("  ✅ blood (血压: 收缩压/舒张压/脉搏)")
        print("  ✅ eeg1, eeg2 (EEG数据路径)")
        print("  ✅ rgb, depth, tobii (多模态数据路径)")
        print("  ✅ audio, video, record (音视频文本JSON)")
        print("  ✅ ptime (时间戳JSON)")
        
    except pymysql.Error as e:
        print(f"\n❌ 数据库错误: {e}")
        print("\n请检查:")
        print("  1. MySQL服务是否启动")
        print("  2. 用户名和密码是否正确")
        print("  3. 用户是否有CREATE DATABASE权限")
        return 1
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(init_database())
