"""
DAG Manager
Manages dynamic directed acyclic graph structure for pipeline visualization
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class DAGManager:
    """
    Manages the DAG structure for interactive visualization
    Tracks main agents and dynamically spawned sub-agents
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.nodes = []
        self.edges = []
        # Use path relative to backend directory
        self.dag_dir = Path(__file__).parent.parent / "dags"
        self.dag_dir.mkdir(parents=True, exist_ok=True)

        # Initialize with main pipeline agents
        self._initialize_main_agents()

    def _initialize_main_agents(self):
        """Initialize DAG with main pipeline agents"""
        main_agents = [
            {
                "id": "data_understanding",
                "type": "main_agent",
                "label": "Data Understanding",
                "parent": None,
                "position": {"x": 100, "y": 50},
                "status": "pending",
                "created_by": None,
                "reason": "Analyze dataset semantics and structure"
            },
            {
                "id": "preprocessing",
                "type": "main_agent",
                "label": "Preprocessing",
                "parent": None,
                "position": {"x": 100, "y": 200},
                "status": "pending",
                "created_by": None,
                "reason": "Clean and prepare data"
            },
            {
                "id": "model_selection",
                "type": "main_agent",
                "label": "Model Selection",
                "parent": None,
                "position": {"x": 100, "y": 350},
                "status": "pending",
                "created_by": None,
                "reason": "Train and compare ML models"
            },
            {
                "id": "hyperparameter_tuning",
                "type": "main_agent",
                "label": "Hyperparameter Tuning",
                "parent": None,
                "position": {"x": 100, "y": 500},
                "status": "pending",
                "created_by": None,
                "reason": "Optimize model parameters"
            },
            {
                "id": "prediction",
                "type": "main_agent",
                "label": "Prediction",
                "parent": None,
                "position": {"x": 100, "y": 650},
                "status": "pending",
                "created_by": None,
                "reason": "Generate predictions"
            }
        ]

        self.nodes = main_agents

        # Create sequential edges between main agents
        self.edges = [
            {"from": "data_understanding", "to": "preprocessing", "type": "sequential"},
            {"from": "preprocessing", "to": "model_selection", "type": "sequential"},
            {"from": "model_selection", "to": "hyperparameter_tuning", "type": "sequential"},
            {"from": "hyperparameter_tuning", "to": "prediction", "type": "sequential"}
        ]

    def add_decision_agent(
        self,
        agent_id: str,
        parent_id: str,
        label: str,
        reason: str,
        agent_type: str = "decision",
        status: str = "running"
    ):
        """
        Add a decision/reasoning agent (like LLM preprocessing planner, RAG agent)

        Args:
            agent_id: Unique identifier
            parent_id: ID of parent stage
            label: Display name
            reason: What this agent decides
            agent_type: Type (decision, rag, analysis)
            status: Current status
        """
        parent_node = next((n for n in self.nodes if n["id"] == parent_id), None)

        if not parent_node:
            print(f"Warning: Parent node {parent_id} not found for agent {agent_id}")
            return

        # Position decision agents to the right of parent
        decision_count = sum(1 for n in self.nodes if n.get("parent") == parent_id and n.get("type") in ["decision", "rag", "analysis"])
        position = {
            "x": parent_node["position"]["x"] + 250,
            "y": parent_node["position"]["y"] - 50 + (decision_count * 80)
        }

        node = {
            "id": agent_id,
            "type": agent_type,
            "label": label,
            "parent": parent_id,
            "position": position,
            "status": status,
            "created_by": parent_id,
            "reason": reason,
            "created_at": datetime.now().isoformat()
        }

        self.nodes.append(node)

        # Create edge
        self.edges.append({
            "from": parent_id,
            "to": agent_id,
            "type": "consults",
            "label": "asks"
        })

        # Create edge from decision agent back to parent (information flow)
        self.edges.append({
            "from": agent_id,
            "to": parent_id,
            "type": "informs",
            "label": "recommends"
        })

        self.save_dag()

    def add_sub_agent(
        self,
        sub_agent_id: str,
        parent_id: str,
        label: str,
        reason: str,
        status: str = "running"
    ):
        """
        Add a dynamically spawned sub-agent to the DAG

        Args:
            sub_agent_id: Unique identifier for sub-agent
            parent_id: ID of parent agent that spawned this sub-agent
            label: Display name for sub-agent
            reason: Why this sub-agent was spawned
            status: Current status (pending, running, completed, failed)
        """
        # Find parent node to position sub-agent relative to it
        parent_node = next((n for n in self.nodes if n["id"] == parent_id), None)

        if not parent_node:
            print(f"Warning: Parent node {parent_id} not found for sub-agent {sub_agent_id}")
            return

        # Calculate position (offset to the right and below parent)
        sub_agent_count = sum(1 for n in self.nodes if n.get("parent") == parent_id and n.get("type") == "sub_agent")
        position = {
            "x": parent_node["position"]["x"] + 250,
            "y": parent_node["position"]["y"] + 60 + (sub_agent_count * 80)
        }

        # Create sub-agent node
        sub_agent_node = {
            "id": sub_agent_id,
            "type": "sub_agent",
            "label": label,
            "parent": parent_id,
            "position": position,
            "status": status,
            "created_by": parent_id,
            "reason": reason,
            "created_at": datetime.now().isoformat()
        }

        self.nodes.append(sub_agent_node)

        # Create edge from parent to sub-agent
        self.edges.append({
            "from": parent_id,
            "to": sub_agent_id,
            "type": "spawned",
            "label": "spawns"
        })

        # Create edge from sub-agent back to parent (results flow)
        self.edges.append({
            "from": sub_agent_id,
            "to": parent_id,
            "type": "reports",
            "label": "reports to"
        })

        # Save updated DAG
        self.save_dag()

    def update_node_status(self, node_id: str, status: str):
        """
        Update the status of a node

        Args:
            node_id: Node identifier
            status: New status (pending, running, completed, failed)
        """
        for node in self.nodes:
            if node["id"] == node_id:
                node["status"] = status
                node["updated_at"] = datetime.now().isoformat()
                break

        self.save_dag()

    def get_dag_structure(self) -> Dict[str, Any]:
        """
        Get the complete DAG structure for visualization

        Returns:
            Dictionary with nodes and edges
        """
        return {
            "job_id": self.job_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "node_count": len(self.nodes),
            "main_agent_count": len([n for n in self.nodes if n["type"] == "main_agent"]),
            "sub_agent_count": len([n for n in self.nodes if n["type"] == "sub_agent"])
        }

    def save_dag(self):
        """Save DAG structure to disk"""
        try:
            filename = self.dag_dir / f"{self.job_id}_dag.json"
            with open(filename, 'w') as f:
                json.dump(self.get_dag_structure(), f, indent=2)
        except Exception as e:
            print(f"Error saving DAG: {e}")

    @classmethod
    def load_dag(cls, job_id: str) -> Optional['DAGManager']:
        """
        Load DAG structure from disk

        Args:
            job_id: Job identifier

        Returns:
            DAGManager instance or None if not found
        """
        try:
            # Use path relative to backend directory
            dag_dir = Path(__file__).parent.parent / "dags"
            filename = dag_dir / f"{job_id}_dag.json"

            if filename.exists():
                with open(filename, 'r') as f:
                    data = json.load(f)

                dag_manager = cls(job_id)
                dag_manager.nodes = data.get("nodes", [])
                dag_manager.edges = data.get("edges", [])
                return dag_manager
        except Exception as e:
            print(f"Error loading DAG: {e}")

        return None

    def get_sub_agents_for_parent(self, parent_id: str) -> List[Dict[str, Any]]:
        """
        Get all sub-agents spawned by a specific parent

        Args:
            parent_id: Parent agent ID

        Returns:
            List of sub-agent nodes
        """
        return [n for n in self.nodes if n.get("parent") == parent_id]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific node by ID"""
        return next((n for n in self.nodes if n["id"] == node_id), None)
