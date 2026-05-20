from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 导入项目配置和基础类
from hvac_fdd.config import get_settings
from hvac_fdd.db.base import Base, make_engine
# 显式导入 orm 模块以确保所有模型都在 Base.metadata 中注册
from hvac_fdd.db import orm

# Alembic Config 对象
config = context.config

# 配置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 核心：将 Base 的元数据关联到迁移目标
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """脱机运行迁移"""
    url = get_settings().database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "pyformat"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """联机运行迁移"""
    # 关键改进：重用项目自身的 make_engine，确保连接池配置一致
    settings = get_settings()
    connectable = make_engine(settings.database_url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
