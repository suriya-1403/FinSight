"""
AgentExecutor manages the execution of the agent's plan.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from finsight.agent.planner import AgentPlanner
from finsight.agent.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Class for executing the agent's plan."""

    def __init__(
            self,
            tool_manager: ToolManager,
            planner: AgentPlanner,
            max_iterations: int = 10,
            max_execution_time: int = 300,  # 5 minutes
    ):
        """
        Initialize the agent executor.

        Args:
            tool_manager: Tool manager for executing tools
            planner: Agent planner for creating and refining plans
            max_iterations: Maximum number of iterations/steps
            max_execution_time: Maximum execution time in seconds
        """
        self.tool_manager = tool_manager
        self.planner = planner
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time

    def execute(self, query: str, context: str = "") -> Dict:
        """
        Execute the agent to answer the query.

        Args:
            query: User query
            context: Optional context

        Returns:
            Execution results
        """
        start_time = time.time()
        execution_log = []
        results = []

        # Step 1: Create the initial plan
        available_tools = self.tool_manager.get_available_tools()
        plan = self.planner.create_plan(query, available_tools, context)

        if not plan or "plan" not in plan or not plan["plan"]:
            logger.warning("Failed to create a valid plan, using fallback approach")
            # Use a simple fallback plan
            plan = {
                "reasoning": "Using fallback plan due to planning failure",
                "plan": [
                    {
                        "step": 1,
                        "description": "Retrieve relevant financial news",
                        "tool": "query_financial_insights",
                        "parameters": {
                            "query": query,
                            "n_results": 5
                        },
                        "purpose": "Find relevant news articles"
                    },
                    {
                        "step": 2,
                        "description": "Generate insights from news",
                        "tool": "generate_insight",
                        "parameters": {
                            "query": query,
                            "relevant_articles": "{{results.0.result}}"
                        },
                        "purpose": "Develop insights from the news"
                    }
                ]
            }

        # Log the initial plan
        execution_log.append({
            "step": "planning",
            "timestamp": time.time(),
            "plan": plan
        })

        # Step 2: Execute the plan
        steps = plan["plan"]
        step_index = 0
        iterations = 0

        while step_index < len(steps) and iterations < self.max_iterations:
            # Check if we've exceeded the maximum execution time
            if time.time() - start_time > self.max_execution_time:
                execution_log.append({
                    "step": "timeout",
                    "timestamp": time.time(),
                    "message": "Execution timed out"
                })
                break

            iterations += 1
            current_step = steps[step_index]

            # Log the current step
            execution_log.append({
                "step": f"execution_{iterations}",
                "timestamp": time.time(),
                "action": current_step
            })

            # Execute the tool for this step
            tool_name = current_step.get("tool")
            parameters = current_step.get("parameters", {})

            # Process parameters, replacing any references to previous results
            processed_parameters = self._process_parameters(parameters, results)

            try:
                tool_result = self.tool_manager.execute_tool(tool_name, processed_parameters)

                # Store the result
                result_entry = {
                    "step_index": step_index,
                    "step": current_step,
                    "result": tool_result
                }
                results.append(result_entry)

                execution_log.append({
                    "step": f"result_{iterations}",
                    "timestamp": time.time(),
                    "result": tool_result
                })

                # Check for errors
                if "error" in tool_result:
                    error = tool_result["error"]
                    logger.warning(f"Error in step {step_index} ({tool_name}): {error}")

                    # Reflect and refine the plan
                    reflection = self.planner.reflect_and_refine(
                        query, plan, results, current_step, error
                    )

                    execution_log.append({
                        "step": f"reflection_{iterations}",
                        "timestamp": time.time(),
                        "reflection": reflection
                    })

                    # Update the plan if needed
                    if reflection.get("updated_plan"):
                        steps = reflection["updated_plan"]
                        plan["plan"] = steps

                    # Decide whether to continue
                    if not reflection.get("should_continue", True):
                        logger.info("Execution stopped based on reflection decision")
                        break

                    # Update the step index
                    next_step = reflection.get("next_step_index", step_index + 1)

                    # Avoid infinite loops
                    if next_step == step_index:
                        logger.warning("Reflection suggested same step again, moving to next step")
                        step_index += 1
                    else:
                        step_index = next_step
                else:
                    # Move to the next step
                    step_index += 1

            except Exception as e:
                error_msg = f"Error executing step {step_index}: {str(e)}"
                logger.error(error_msg)

                execution_log.append({
                    "step": f"error_{iterations}",
                    "timestamp": time.time(),
                    "error": error_msg
                })

                # Add error as result
                result_entry = {
                    "step_index": step_index,
                    "step": current_step,
                    "error": error_msg
                }
                results.append(result_entry)

                # Move to next step despite error
                step_index += 1

        # Step 3: Generate the final answer
        final_answer = self._generate_answer(query, results, plan)

        execution_log.append({
            "step": "answer_generation",
            "timestamp": time.time(),
            "answer": final_answer
        })

        return {
            "query": query,
            "plan": plan,
            "results": results,
            "answer": final_answer,
            "execution_log": execution_log,
            "execution_time": time.time() - start_time
        }

    def _process_parameters(self, parameters, results):
        """
        Process parameters to replace references to previous results.

        Args:
            parameters: Original parameters
            results: Previous execution results

        Returns:
            Processed parameters
        """
        processed = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # This is a reference to a previous result
                try:
                    # Extract the reference path
                    ref_path = value[2:-2].strip()
                    parts = ref_path.split(".")

                    if parts[0] == "results":
                        # Reference to a specific result
                        result_index = int(parts[1])
                        if result_index < len(results):
                            result_obj = results[result_index]

                            # Navigate to the specific field
                            field_value = result_obj
                            for part in parts[2:]:
                                if isinstance(field_value, dict) and part in field_value:
                                    field_value = field_value[part]
                                else:
                                    field_value = None
                                    break

                            processed[key] = field_value
                        else:
                            logger.warning(f"Reference to non-existent result: {ref_path}")
                            processed[key] = None
                    else:
                        logger.warning(f"Unsupported reference type: {ref_path}")
                        processed[key] = value
                except Exception as e:
                    logger.error(f"Error processing parameter reference {value}: {str(e)}")
                    processed[key] = value
            else:
                # Regular value
                processed[key] = value

        return processed

    def _generate_answer(self, query: str, results: List[Dict], plan: Dict) -> Dict:
        """
        Generate the final answer based on the results.

        Args:
            query: User query
            results: Execution results
            plan: Execution plan

        Returns:
            Final answer
        """
        # Initialize the answer object
        answer = {
            "query": query,
            "summary": "",
            "insights": [],
            "visualizations": [],
            "data_points": [],
            "confidence": 0.0
        }

        # Extract insights from sentiment analysis results
        for result in results:
            tool_result = result.get("result", {})

            # Handle specific tool results
            if "tool" in tool_result:
                tool_name = tool_result["tool"]

                # Extract sentiment analysis results
                if tool_name == "news_sentiment_agent" and "result" in tool_result:
                    sentiment_result = tool_result["result"]

                    # Extract overall sentiment
                    if isinstance(sentiment_result, dict):
                        overall_sentiment = sentiment_result.get("overall_sentiment", "neutral")
                        confidence = sentiment_result.get("confidence", 0.0)

                        answer["insights"].append({
                            "type": "sentiment",
                            "value": overall_sentiment,
                            "confidence": confidence
                        })

                        # Extract key factors
                        if "key_factors" in sentiment_result:
                            key_factors = sentiment_result["key_factors"]

                            if "positive" in key_factors:
                                for factor in key_factors["positive"]:
                                    answer["insights"].append({
                                        "type": "positive_factor",
                                        "value": factor
                                    })

                            if "negative" in key_factors:
                                for factor in key_factors["negative"]:
                                    answer["insights"].append({
                                        "type": "negative_factor",
                                        "value": factor
                                    })

                # Extract technical analysis results
                elif tool_name == "technical_analysis_agent" and "result" in tool_result:
                    tech_result = tool_result["result"]

                    if isinstance(tech_result, dict):
                        trend = tech_result.get("trend_direction", "sideways")
                        recommendation = tech_result.get("recommendation", "hold")

                        answer["insights"].append({
                            "type": "trend",
                            "value": trend
                        })

                        answer["insights"].append({
                            "type": "recommendation",
                            "value": recommendation,
                            "timeframe": tech_result.get("recommendation_timeframe", "medium")
                        })

                # Extract price forecast results
                elif tool_name == "forecast_prices" and "result" in tool_result:
                    forecast_result = tool_result["result"]

                    if isinstance(forecast_result, dict) and "forecast" in forecast_result:
                        forecast = forecast_result["forecast"]

                        if forecast and len(forecast) > 0:
                            last_price = forecast_result.get("last_known_price", 0)
                            last_forecast = forecast[-1]

                            if last_price > 0 and last_forecast > 0:
                                change_pct = ((last_forecast / last_price) - 1) * 100

                                answer["insights"].append({
                                    "type": "price_forecast",
                                    "value": {
                                        "current_price": last_price,
                                        "forecasted_price": last_forecast,
                                        "change_percentage": change_pct
                                    }
                                })

                # Extract visualization results
                elif tool_name in ["create_price_chart", "create_sentiment_chart",
                                   "create_technical_dashboard"] and "result" in tool_result:
                    visualization = tool_result["result"]

                    if visualization:
                        answer["visualizations"].append({
                            "type": tool_name,
                            "data": visualization
                        })

                # Extract generated insights
                elif tool_name == "generate_insight" and "result" in tool_result:
                    insight_result = tool_result["result"]

                    if isinstance(insight_result, dict) and "insight" in insight_result:
                        answer["insights"].append({
                            "type": "generated_insight",
                            "value": insight_result["insight"]
                        })

                        # Extract sources if available
                        if "sources" in insight_result:
                            for source in insight_result["sources"]:
                                answer["data_points"].append({
                                    "type": "source",
                                    "value": source
                                })

        # Generate a summary based on collected insights
        answer["summary"] = self._generate_summary(answer["insights"])

        # Calculate overall confidence
        confidence_values = [insight.get("confidence", 0.0) for insight in answer["insights"]
                             if "confidence" in insight]

        if confidence_values:
            answer["confidence"] = sum(confidence_values) / len(confidence_values)

        return answer

    def _generate_summary(self, insights: List[Dict]) -> str:
        """
        Generate a summary based on the insights.

        Args:
            insights: List of insights

        Returns:
            Summary text
        """
        # Simplified summary generation based on insights
        summary = "Based on the analysis:\n\n"

        # Sentiment
        sentiment_insights = [i for i in insights if i.get("type") == "sentiment"]
        if sentiment_insights:
            sentiment = sentiment_insights[0].get("value", "neutral")
            confidence = sentiment_insights[0].get("confidence", 0.0)
            summary += f"- Market sentiment appears to be {sentiment.upper()} "
            summary += f"(confidence: {confidence:.2f})\n"

        # Positive factors
        positive_factors = [i.get("value") for i in insights if i.get("type") == "positive_factor"]
        if positive_factors:
            summary += "- Positive factors include: " + ", ".join(positive_factors[:3]) + "\n"

        # Negative factors
        negative_factors = [i.get("value") for i in insights if i.get("type") == "negative_factor"]
        if negative_factors:
            summary += "- Negative factors include: " + ", ".join(negative_factors[:3]) + "\n"

        # Trend
        trend_insights = [i for i in insights if i.get("type") == "trend"]
        if trend_insights:
            trend = trend_insights[0].get("value", "sideways")
            summary += f"- Market trend is {trend.upper()}\n"

        # Recommendation
        recommendation_insights = [i for i in insights if i.get("type") == "recommendation"]
        if recommendation_insights:
            recommendation = recommendation_insights[0].get("value", "hold")
            timeframe = recommendation_insights[0].get("timeframe", "medium")
            summary += f"- Technical analysis suggests to {recommendation.upper()} "
            summary += f"for {timeframe}-term\n"

        # Price forecast
        forecast_insights = [i for i in insights if i.get("type") == "price_forecast"]
        if forecast_insights:
            forecast_data = forecast_insights[0].get("value", {})
            current_price = forecast_data.get("current_price", 0)
            forecasted_price = forecast_data.get("forecasted_price", 0)
            change_pct = forecast_data.get("change_percentage", 0)

            summary += f"- Price forecast: ${current_price:.2f} → ${forecasted_price:.2f} "
            summary += f"({change_pct:+.2f}%)\n"

        # Add generated insights if available
        generated_insights = [i.get("value") for i in insights if i.get("type") == "generated_insight"]
        if generated_insights:
            summary += "\n" + generated_insights[0]

        return summary