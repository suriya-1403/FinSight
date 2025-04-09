# finsight/__init__.py
"""
FinSight AI - Financial insights from news and market data.

This package provides tools for collecting, analyzing, and generating
insights from financial news and market data.
"""

import logging
# Import only constants at the module level to avoid circular dependencies
from finsight.config.settings import DEFAULT_NEWS_QUERIES, DEFAULT_TICKERS, DEFAULT_DAYS_BACK

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

__version__ = "0.1.0"

def setup_finsight():
    """
    Initialize and expose the main components of the finsight package.
    
    This function performs lazy loading of components to prevent circular imports.
    Call this function when you need access to the main classes like AgenticRAGController,
    AgentPlanner, etc.
    
    Returns:
        dict: A dictionary containing all the main components of the finsight package.
    """
    # Import components only when this function is called
    from finsight.agent.controller import AgenticRAGController
    from finsight.agent.planner import AgentPlanner
    from finsight.agent.tool_manager import ToolManager
    from finsight.agent.executor import AgentExecutor
    from finsight.pipeline.pipeline_manager import PipelineManager
    from finsight.master import FinSightMaster
    
    return {
        'AgenticRAGController': AgenticRAGController,
        'AgentPlanner': AgentPlanner,
        'ToolManager': ToolManager,
        'AgentExecutor': AgentExecutor,
        'PipelineManager': PipelineManager,
        'FinSightMaster': FinSightMaster,
    }

__all__ = [
    'setup_finsight',
    'DEFAULT_NEWS_QUERIES',
    'DEFAULT_TICKERS',
    'DEFAULT_DAYS_BACK',
    '__version__'
]
