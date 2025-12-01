"""
Agent Chat Handler
Enables conversational interaction with pipeline agents
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import get_openai_config


class AgentChatHandler:
    """
    Handles chat interactions with individual pipeline agents
    """

    def __init__(self):
        config = get_openai_config()
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=0.3,  # Slightly higher for more conversational responses
            api_key=config['api_key']
        )
        # Use path relative to backend directory
        self.conversations_dir = Path(__file__).parent.parent / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def chat_with_agent(
        self,
        job_id: str,
        agent_id: str,
        user_message: str,
        agent_state: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message to a specific agent

        Args:
            job_id: Pipeline job identifier
            agent_id: Which agent to chat with (data_understanding, preprocessing, etc.)
            user_message: User's question or message
            agent_state: Current state/results of the agent
            conversation_history: Previous conversation messages

        Returns:
            Dictionary with response and updated conversation
        """
        if conversation_history is None:
            conversation_history = []

        # Get agent-specific context
        agent_context = self._get_agent_context(agent_id, agent_state)

        # Build prompt with agent persona
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_agent_persona(agent_id)),
            ("system", f"Agent Context:\n{agent_context}"),
            *self._format_conversation_history(conversation_history),
            ("user", "{user_message}")
        ])

        try:
            # Get LLM response
            response = self.llm.invoke(
                prompt.format_messages(user_message=user_message)
            )
            assistant_message = response.content

            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Save conversation
            self._save_conversation(job_id, agent_id, conversation_history)

            # Extract references if any
            references = self._extract_references(assistant_message, agent_state)

            return {
                "success": True,
                "response": assistant_message,
                "conversation_history": conversation_history,
                "references": references,
                "suggestions": self._generate_suggestions(agent_id, user_message)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error processing your message: {str(e)}"
            }

    def _get_agent_persona(self, agent_id: str) -> str:
        """Get the personality/role description for each agent"""
        personas = {
            "data_understanding": """You are a Data Understanding Teacher - an expert who LOVES showing students exactly what you found in THEIR data.

YOUR TEACHING STYLE:
- Always use SPECIFIC EXAMPLES from the actual data (column names, values, numbers)
- Explain concepts by pointing to what you saw: "For example, I noticed your 'age' column had..."
- Use analogies and simple language
- When discussing patterns, give concrete examples: "Look at row 5 where..."
- NEVER give generic answers - always tie back to THIS specific dataset

REMEMBER: The user wants to LEARN, so teach them by showing, not just telling!""",

            "preprocessing": """You are a Preprocessing Teacher - you explain data cleaning like a patient instructor showing a student EXACTLY what you did and WHY.

YOUR TEACHING STYLE:
- Point to SPECIFIC columns you changed: "I found 3 missing values in your 'salary' column..."
- Explain your decisions with data evidence: "I chose median over mean because your data had outliers - see how..."
- Use before/after examples: "Before: column X had [values], After: [values]"
- Explain trade-offs: "I could have dropped the rows, but that would lose 10% of your data..."
- Show the LLM reasoning: "The AI planner suggested this because..."

Make them understand not just WHAT you did, but WHY it was the right choice for THEIR data!""",

            "model_selection": """You are a Model Selection Teacher - you explain model performance by showing EXACTLY how each model performed on THEIR data.

YOUR TEACHING STYLE:
- Compare models with actual scores: "RandomForest got 0.95, while LogisticRegression got 0.87..."
- Explain why in context of their data: "RandomForest won because your data has non-linear patterns..."
- Use simple analogies: "Think of it like..."
- Point to specific features: "The 'income' feature was most important because..."

Teach them to UNDERSTAND model behavior, not just accept results!""",

            "hyperparameter_tuning": """You are a Hyperparameter Tuning Teacher - you explain optimization by showing the journey from starting parameters to optimal ones.

YOUR TEACHING STYLE:
- Show the improvement: "Started at 0.85, ended at 0.92 - a 7% improvement!"
- Explain parameter choices: "I tried max_depth from 3 to 10, and 7 worked best because..."
- Use analogies: "Think of max_depth like how many questions you can ask..."
- Show what didn't work and why

Make them understand the tuning process as exploration, not magic!""",

            "prediction": """You are a Prediction Teacher - you explain predictions by showing examples and building intuition.

YOUR TEACHING STYLE:
- Give specific prediction examples: "For row 10, I predicted..."
- Explain confidence: "I'm 95% confident because..."
- Point to which features drove the prediction
- Show edge cases: "This prediction is uncertain because..."

Help them trust AND question the model appropriately!"""
        }

        return personas.get(agent_id, "You are a helpful AI assistant for data science tasks.")

    def _get_agent_context(self, agent_id: str, agent_state: Dict[str, Any]) -> str:
        """Extract detailed, actionable context from agent state"""
        if not agent_state or "data" not in agent_state:
            return "No context available yet - agent hasn't run."

        context_parts = []
        data = agent_state.get("data", {})

        # Preprocessing context - DETAILED
        if agent_id == "preprocessing":
            context_parts.append("=== PREPROCESSING ACTIONS I PERFORMED ===")
            context_parts.append(f"Original Dataset: {data.get('original_shape', 'N/A')} rows x columns")
            context_parts.append(f"Final Dataset: {data.get('final_shape', 'N/A')} rows x columns")

            if "steps_performed" in data:
                context_parts.append("\n=== DETAILED PREPROCESSING STEPS ===")
                for step in data["steps_performed"]:
                    step_name = step.get('step', 'Unknown')
                    action = step.get('action', 'N/A')
                    context_parts.append(f"\n{step_name}:")
                    context_parts.append(f"  Action: {action}")

                    # Add step-specific details
                    if "columns_affected" in step:
                        context_parts.append(f"  Columns: {', '.join(step['columns_affected'][:5])}")
                    if "missing_before" in step:
                        context_parts.append(f"  Missing values before: {step['missing_before']}")
                        context_parts.append(f"  Missing values after: {step['missing_after']}")
                    if "outliers_capped" in step:
                        context_parts.append(f"  Outliers capped: {step['outliers_capped']}")
                    if "duplicates_removed" in step:
                        context_parts.append(f"  Duplicates removed: {step['duplicates_removed']}")
                    if "encoding_details" in step:
                        context_parts.append(f"  Encoding applied to {len(step['encoding_details'])} columns")
                    if "scaler_type" in step:
                        context_parts.append(f"  Scaler: {step['scaler_type']}")
                    if "columns_removed" in step:
                        context_parts.append(f"  Removed columns: {step['columns_removed']}")

            # RAG recommendations
            if "rag_recommendations" in data:
                rag = data["rag_recommendations"]
                if "explanation" in rag:
                    context_parts.append(f"\n=== RAG INSIGHTS ===")
                    context_parts.append(rag["explanation"][:500])  # First 500 chars

            # Sub-agents spawned
            if "sub_agents" in data:
                sub_agents = data["sub_agents"]
                if sub_agents.get("spawned"):
                    context_parts.append(f"\n=== SUB-AGENTS SPAWNED ===")
                    context_parts.append(f"I spawned {sub_agents['count']} specialized sub-agent(s):")
                    for sa in sub_agents.get("outputs", []):
                        context_parts.append(f"\n- {sa.get('sub_agent', 'Unknown')}:")
                        if "llm_recommendation" in sa:
                            context_parts.append(f"  {sa['llm_recommendation'][:300]}")

            # LLM Reasoning (from Intelligent Preprocessing Agent)
            if "llm_reasoning" in data:
                context_parts.append(f"\n=== WHY I MADE THESE CHOICES (LLM REASONING) ===")
                context_parts.append(data["llm_reasoning"])

            if "llm_preprocessing_plan" in data:
                plan = data["llm_preprocessing_plan"]
                context_parts.append(f"\n=== MY PREPROCESSING STRATEGY ===")
                for key, value in plan.items():
                    if key != "reasoning":
                        context_parts.append(f"  {key}: {value}")

            # Summary
            if "summary" in data:
                context_parts.append(f"\n=== SUMMARY ===")
                context_parts.append(data["summary"])

        # Data Understanding context - DETAILED
        elif agent_id == "data_understanding":
            context_parts.append("=== DATA ANALYSIS I PERFORMED ===")

            if "basic_info" in data:
                info = data["basic_info"]
                context_parts.append(f"Rows: {info.get('num_rows', 'N/A')}")
                context_parts.append(f"Columns: {info.get('num_columns', 'N/A')}")
                context_parts.append(f"Memory Usage: {info.get('memory_usage', 'N/A')}")

            if "problem_type" in data:
                pt = data["problem_type"]
                context_parts.append(f"\n=== PROBLEM TYPE IDENTIFIED ===")
                context_parts.append(f"Type: {pt.get('type', 'N/A')}")
                context_parts.append(f"Target Column: {pt.get('suggested_target_column', 'N/A')}")
                context_parts.append(f"Reasoning: {pt.get('reasoning', 'N/A')}")

            if "data_quality" in data:
                dq = data["data_quality"]
                context_parts.append(f"\n=== DATA QUALITY ASSESSMENT ===")
                context_parts.append(f"Overall Score: {dq.get('quality_score', 'N/A')}/100")
                context_parts.append(f"Missing Values: {dq.get('missing_percentage', 'N/A')}%")
                if "issues" in dq:
                    context_parts.append(f"Issues Found: {', '.join(dq['issues'][:3])}")

            if "column_analysis" in data:
                context_parts.append(f"\n=== COLUMNS ANALYZED ===")
                context_parts.append(f"Analyzed {len(data['column_analysis'])} columns")

        # Model Selection context - DETAILED
        elif agent_id == "model_selection":
            if "models_trained" in data:
                context_parts.append("=== MODELS I TRAINED AND EVALUATED ===")
                for model in data["models_trained"][:5]:  # Top 5
                    context_parts.append(f"\n{model.get('model_name', 'Unknown')}:")
                    context_parts.append(f"  Score: {model.get('score', 'N/A')}")
                    if "params" in model:
                        context_parts.append(f"  Parameters: {model['params']}")

            if "best_model" in data:
                best = data["best_model"]
                context_parts.append(f"\n=== BEST MODEL SELECTED ===")
                context_parts.append(f"Model: {best.get('model_name', 'N/A')}")
                context_parts.append(f"Score: {best.get('score', 'N/A')}")
                context_parts.append(f"Why: {best.get('reason', 'Highest performance')}")

        # Hyperparameter Tuning context - DETAILED
        elif agent_id == "hyperparameter_tuning":
            context_parts.append("=== HYPERPARAMETER TUNING I PERFORMED ===")
            if "best_params" in data:
                context_parts.append(f"Optimal Parameters Found:")
                for param, value in list(data["best_params"].items())[:10]:
                    context_parts.append(f"  {param}: {value}")

            if "tuning_results" in data:
                context_parts.append(f"\nTuning Iterations: {len(data['tuning_results'])}")

        return "\n".join(context_parts) if context_parts else "Agent state available but no specific context extracted."

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> List[tuple]:
        """Format conversation history for prompt"""
        formatted = []
        for msg in history[-6:]:  # Last 6 messages (3 exchanges)
            role = "user" if msg["role"] == "user" else "assistant"
            formatted.append((role, msg["content"]))
        return formatted

    def _extract_references(self, response: str, agent_state: Dict[str, Any]) -> List[str]:
        """Extract references to specific data points or visualizations"""
        references = []

        # Look for visualization references
        if "visualizations" in agent_state:
            viz_paths = agent_state.get("visualizations", [])
            for viz in viz_paths:
                if any(keyword in response.lower() for keyword in ["chart", "plot", "graph", "visualization"]):
                    references.append(viz)

        return references

    def _generate_suggestions(self, agent_id: str, user_message: str) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions_map = {
            "data_understanding": [
                "What does the target column distribution look like?",
                "Are there any missing values I should be concerned about?",
                "What feature relationships are most important?"
            ],
            "preprocessing": [
                "Why did you choose this imputation strategy?",
                "How many outliers were detected?",
                "What encoding method was used for categorical variables?"
            ],
            "model_selection": [
                "Why did this model perform best?",
                "What are the key performance metrics?",
                "How do the models compare?"
            ],
            "hyperparameter_tuning": [
                "What parameters were tuned?",
                "How much did performance improve?",
                "What was the tuning strategy?"
            ],
            "prediction": [
                "What's the confidence level of predictions?",
                "How reliable are these predictions?",
                "What features influenced predictions most?"
            ]
        }

        return suggestions_map.get(agent_id, [])

    def _save_conversation(self, job_id: str, agent_id: str, conversation: List[Dict[str, str]]):
        """Save conversation to disk"""
        try:
            filename = self.conversations_dir / f"{job_id}_{agent_id}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "job_id": job_id,
                    "agent_id": agent_id,
                    "conversation": conversation
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self, job_id: str, agent_id: str) -> List[Dict[str, str]]:
        """Load conversation history from disk"""
        try:
            filename = self.conversations_dir / f"{job_id}_{agent_id}.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
                    return data.get("conversation", [])
        except Exception as e:
            print(f"Error loading conversation: {e}")

        return []
