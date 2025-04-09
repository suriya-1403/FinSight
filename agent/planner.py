"""
AgentPlanner manages the planning and reasoning aspects of the agentic RAG system.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


class AgentPlanner:
    """Class for planning and reasoning in the agentic RAG system."""

    def __init__(
            self,
            model: str = "llama3.2",
            base_url: str = "http://localhost:11434",
    ):
        """Initialize the planner with an LLM."""
        self.model_name = model
        self.base_url = base_url

        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=model, base_url=base_url)
            logger.info(f"Initialized AgentPlanner with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing LLM for AgentPlanner: {str(e)}")
            self.llm = None

        # Create prompt templates
        self.planning_prompt = PromptTemplate(
            input_variables=["query", "available_tools", "context"],
            template="""You are an intelligent financial analysis agent. Given a user query, plan the steps needed to provide a comprehensive answer.

USER QUERY: {query}

AVAILABLE TOOLS:
{available_tools}

CONTEXT (if any):
{context}

First, analyze what information is needed to answer this query comprehensively.
Then, create a detailed plan with specific steps to gather and analyze that information.

Your plan should include:
1. What specific information needs to be retrieved
2. Which tools to use for each retrieval step
3. What analysis should be performed on the retrieved information
4. How to synthesize the information into a comprehensive answer

Return your plan as a JSON object with the following structure:
```json
{{
    "reasoning": "Your step-by-step reasoning about how to approach this query",
    "plan": [
        {{
            "step": 1,
            "description": "Brief description of this step",
            "tool": "Name of the tool to use",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "purpose": "Why this step is necessary"
        }},
        ...
    ],
    "fallback_strategy": "What to do if the plan cannot be executed",
    "limitations": ["Any limitations in the current approach"]
}}
```
"""
        )

        self.reflection_prompt = PromptTemplate(
            input_variables=["query", "plan", "results_so_far", "current_step", "error"],
            template="""You are an intelligent financial analysis agent. Reflect on the current state of your plan execution and determine how to proceed.

USER QUERY: {query}

ORIGINAL PLAN:
{plan}

RESULTS SO FAR:
{results_so_far}

CURRENT STEP:
{current_step}

ERROR (if any):
{error}

Reflect on what has happened so far and decide how to adjust your plan. Consider:
1. Was the information retrieved what you expected?
2. Did any steps fail, and if so, why?
3. Is the current approach still viable, or do you need to change strategy?
4. Are there any new insights that should inform your next steps?

Return your reflection and updated plan as a JSON object:
```json
{{
    "reflection": "Your detailed thoughts on the current state and what to do next",
    "updated_plan": [
        {{
            "step": 1,
            "description": "Brief description of this step",
            "tool": "Name of the tool to use",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "purpose": "Why this step is necessary"
        }},
        ...
    ],
    "should_continue": true or false,
    "next_step_index": 0-based index of the next step to execute
}}
```
"""
        )

        # Create LLM chains
        if self.llm:
            self.planning_chain = LLMChain(llm=self.llm, prompt=self.planning_prompt)
            self.reflection_chain = LLMChain(llm=self.llm, prompt=self.reflection_prompt)
        else:
            self.planning_chain = None
            self.reflection_chain = None

    """
    Add this method to the AgentPlanner class to create more robust default plans.
    """

    def create_plan(self, query: str, available_tools: Dict, context: str = "") -> Dict:
        """
        Create a plan to answer the user's query.

        Args:
            query: The user's query
            available_tools: Dictionary of available tools with descriptions
            context: Any context information to inform planning

        Returns:
            Dictionary containing the plan
        """
        if not self.planning_chain:
            return self._fallback_plan(query)

        try:
            # Format the available tools for the prompt
            tools_text = "\n".join([f"- {name}: {desc}" for name, desc in available_tools.items()])

            # Generate the plan
            response = self.planning_chain.invoke({
                "query": query,
                "available_tools": tools_text,
                "context": context
            })

            # Extract and parse the plan
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response)

            # Extract JSON from the response
            import re
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    plan = json.loads(json_str)

                    # Validate plan structure
                    if "plan" not in plan or not isinstance(plan["plan"], list) or len(plan["plan"]) == 0:
                        logger.warning("Invalid plan structure, using fallback")
                        return self._fallback_plan(query)

                    # Validate and fix each step
                    for i, step in enumerate(plan["plan"]):
                        # Ensure required fields
                        if "tool" not in step:
                            logger.warning(f"Step {i + 1} missing tool, skipping")
                            continue

                        # Ensure parameters
                        if "parameters" not in step or not isinstance(step["parameters"], dict):
                            step["parameters"] = {}

                        # Ensure query parameter for relevant tools
                        if step["tool"] in ["query_financial_insights", "news_sentiment_agent",
                                            "technical_analysis_agent", "generate_insight"]:
                            if "query" not in step["parameters"]:
                                step["parameters"]["query"] = query
                                logger.info(f"Added missing query parameter to step {i + 1}")

                        # Ensure ticker parameter for forecast_prices
                        if step["tool"] == "forecast_prices" and "ticker" not in step["parameters"]:
                            # Extract ticker from query or use default
                            ticker = self._extract_ticker_from_query(query) or "TSLA"
                            step["parameters"]["ticker"] = ticker
                            logger.info(f"Added missing ticker parameter to step {i + 1}")

                    logger.info(f"Successfully created plan with {len(plan.get('plan', []))} steps")
                    return plan
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing plan JSON: {str(e)}")
                    return self._fallback_plan(query)
            else:
                logger.error("No JSON found in planning response")
                return self._fallback_plan(query)

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            return self._fallback_plan(query)

    def _extract_ticker_from_query(self, query: str) -> str:
        """Extract a ticker symbol from the query."""
        import re

        # Look for ticker patterns
        ticker_match = re.search(r'\b[A-Z]{1,5}-?USD?\b', query)
        if ticker_match:
            return ticker_match.group(0)

        # Common company name mappings
        company_map = {
            'tesla': 'TSLA',
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'facebook': 'META',
            'meta': 'META',
            'netflix': 'NFLX',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD'
        }

        query_lower = query.lower()
        for company, ticker in company_map.items():
            if company in query_lower:
                return ticker

        # Default to a common ticker if nothing found
        if 'stock' in query_lower:
            return 'SPY'  # S&P 500 ETF
        elif 'crypto' in query_lower or 'bitcoin' in query_lower:
            return 'BTC-USD'

        return None

    def _fallback_plan(self, query: str) -> Dict:
        """Create a fallback plan when planning fails."""
        # Extract ticker from query if present
        ticker = self._extract_ticker_from_query(query) or "TSLA"

        return {
            "reasoning": "Fallback planning due to LLM unavailability or planning failure",
            "plan": [
                {
                    "step": 1,
                    "description": "Retrieve relevant financial news about the query",
                    "tool": "query_financial_insights",
                    "parameters": {
                        "query": query,
                        "n_results": 10
                    },
                    "purpose": "Get recent financial news relevant to the query"
                },
                {
                    "step": 2,
                    "description": "Collect market data for relevant ticker",
                    "tool": "collect_market_data",
                    "parameters": {
                        "tickers": [ticker],
                        "period": "60d"
                    },
                    "purpose": "Get price and volume data to analyze performance"
                },
                {
                    "step": 3,
                    "description": "Analyze sentiment in the retrieved news",
                    "tool": "news_sentiment_agent",
                    "parameters": {
                        "query": query,
                        "news_articles": "results from step 1"
                    },
                    "purpose": "Understand market sentiment"
                },
                {
                    "step": 4,
                    "description": "Perform technical analysis on the market data",
                    "tool": "technical_analysis_agent",
                    "parameters": {
                        "query": query,
                        "market_data": "results from step 2"
                    },
                    "purpose": "Analyze price patterns and indicators"
                },
                {
                    "step": 5,
                    "description": "Generate insights based on all analysis",
                    "tool": "generate_insight",
                    "parameters": {
                        "query": query,
                        "relevant_articles": "results from step 1"
                    },
                    "purpose": "Create a comprehensive analysis of performance"
                }
            ],
            "fallback_strategy": "Use predefined retrieval and analysis steps",
            "limitations": ["This is a generic plan that may not be optimized for the specific query"]
        }

    def reflect_and_refine(
            self,
            query: str,
            original_plan: Dict,
            results_so_far: List[Dict],
            current_step: Dict,
            error: str = ""
    ) -> Dict:
        """
        Reflect on the execution progress and refine the plan if needed.

        Args:
            query: The original user query
            original_plan: The original plan
            results_so_far: Results from executed steps
            current_step: The current step that was executed
            error: Any error that occurred during execution

        Returns:
            Dictionary with reflection and updated plan
        """
        if not self.reflection_chain:
            return {
                "reflection": "Cannot perform reflection due to unavailable LLM",
                "updated_plan": original_plan.get("plan", []),
                "should_continue": True,
                "next_step_index": 0
            }

        try:
            # Format the inputs - ensuring they're serializable
            plan_text = self._prepare_serializable_json(original_plan)

            # Prepare serializable results
            serializable_results = []
            for result in results_so_far:
                serializable_results.append(self._prepare_serializable_json(result))

            results_text = json.dumps(serializable_results, indent=2)
            current_step_text = json.dumps(self._prepare_serializable_json(current_step), indent=2)

            # Generate the reflection
            response = self.reflection_chain.invoke({
                "query": query,
                "plan": plan_text,
                "results_so_far": results_text,
                "current_step": current_step_text,
                "error": error
            })

            # Extract and parse the reflection
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response)

            # Extract JSON from the response
            import re
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    reflection = json.loads(json_str)
                    logger.info("Successfully reflected and refined plan")
                    return reflection
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing reflection JSON: {str(e)}")
            else:
                logger.error("No JSON found in reflection response")

            # Fallback reflection
            return {
                "reflection": "Unable to properly reflect on the plan execution",
                "updated_plan": original_plan.get("plan", []),
                "should_continue": True,
                "next_step_index": len(results_so_far)
            }

        except Exception as e:
            logger.error(f"Error in reflection: {str(e)}")
            return {
                "reflection": f"Error in reflection: {str(e)}",
                "updated_plan": original_plan.get("plan", []),
                "should_continue": True,
                "next_step_index": len(results_so_far)
            }

    def _prepare_serializable_json(self, obj):
        """Make an object JSON serializable by handling special types."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[k] = self._prepare_serializable_json(v)
            return result
        elif isinstance(obj, list):
            return [self._prepare_serializable_json(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            # Handle pandas DataFrame or Series
            try:
                return {
                    "type": "dataframe",
                    "summary": f"DataFrame with shape {obj.shape if hasattr(obj, 'shape') else 'unknown'}"
                }
            except:
                return str(obj)
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle other non-serializable objects
            return str(obj)
        else:
            return obj

