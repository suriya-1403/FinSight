"""
Multi-agent system for financial analysis.
"""

from finsight.agents.base_agents import (
    BaseAgent,
    NewsSentimentAgent,
    # TechnicalAnalysisAgent,
    # FundamentalAnalysisAgent,
    # RiskAssessmentAgent,
    # AgentController
)

from finsight.agents.agent_controller import AgentSystem

__all__ = [
    'BaseAgent',
    'NewsSentimentAgent',
    # 'TechnicalAnalysisAgent',
    # 'FundamentalAnalysisAgent',
    # 'RiskAssessmentAgent',
    # 'AgentController',
    'AgentSystem'
]