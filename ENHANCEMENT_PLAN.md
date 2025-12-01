# Data Monkey Enhancement Plan
## Interactive DAG, Adaptive Agents, RAG, and Chat Interface

### Overview
Transform Data Monkey from a linear pipeline into an intelligent, adaptive system with:
1. Interactive DAG visualization
2. Self-improving agents that can spawn sub-agents
3. RAG-powered knowledge retrieval
4. Chat interface for each agent

---

## 1. Interactive DAG Visualization

### Frontend (React Flow)
- **Library**: React Flow for interactive graph rendering
- **Features**:
  - Drag nodes to rearrange
  - Zoom and pan
  - Click nodes to expand/collapse details
  - Real-time updates as agents spawn/complete
  - Color coding by status
  - Animation for active nodes

### Backend Changes
- **New Endpoint**: `GET /api/pipeline/dag/{job_id}`
  - Returns full DAG structure with all nodes and edges
  - Includes dynamically created sub-agents

- **DAG State Management**:
  ```python
  {
    "nodes": [
      {
        "id": "agent_1",
        "type": "main_agent",  # or "sub_agent"
        "label": "Data Understanding",
        "parent": null,  # or parent_id for sub-agents
        "position": {"x": 0, "y": 0},
        "status": "completed",
        "created_by": null,  # or parent agent id
        "reason": "Initial pipeline stage"
      }
    ],
    "edges": [
      {"from": "agent_1", "to": "agent_2", "type": "sequential"},
      {"from": "agent_2", "to": "agent_2_sub_1", "type": "spawned"}
    ]
  }
  ```

---

## 2. Adaptive Agent System

### Agent Self-Improvement Loop

Each agent follows this pattern:
```
1. Execute main task
2. Analyze results/statistics
3. Decide if additional processing needed
4. If yes: Spawn specialized sub-agent
5. Integrate sub-agent results
6. Update pipeline state
```

### Sub-Agent Types

**For Preprocessing Agent**:
- `OutlierAnalysisAgent` - Deep dive into outlier patterns
- `FeatureEngineeringAgent` - Create derived features
- `DataBalancingAgent` - Handle imbalanced datasets
- `MissingDataStrategyAgent` - Advanced imputation strategies

**Decision Logic**:
```python
class AdaptivePreprocessingAgent:
    def execute(self, data):
        results = self.preprocess(data)

        # Analyze results
        insights = self.analyze_preprocessing_quality(results)

        # Decide on sub-agents
        if insights['outlier_percentage'] > 10:
            sub_agent = OutlierAnalysisAgent()
            outlier_strategy = sub_agent.recommend_strategy(data)
            results = self.apply_outlier_strategy(outlier_strategy)

        if insights['missing_data_complex']:
            sub_agent = MissingDataStrategyAgent()
            imputation_method = sub_agent.advanced_impute(data)
            results = self.apply_imputation(imputation_method)

        return results
```

### Implementation Files
- `backend/agents/adaptive_agent_base.py` - Base class with spawning logic
- `backend/agents/sub_agents/` - Directory for specialized sub-agents
- `backend/pipeline/dag_manager.py` - Manages dynamic DAG updates

---

## 3. RAG Agent for Best Practices

### Knowledge Base
Store preprocessing best practices in vector database:

**Knowledge Sources**:
1. Scikit-learn documentation
2. Data preprocessing research papers
3. Industry best practices
4. Common data science patterns

**Vector Store**: Use Chroma or FAISS for local storage

### RAG Agent Implementation

```python
class RAGPreprocessingAgent:
    def __init__(self):
        self.vector_store = self.load_knowledge_base()
        self.llm = ChatOpenAI()

    def get_recommendations(self, data_analysis):
        # Create query from data characteristics
        query = f"""
        Dataset has:
        - {data_analysis['missing_percentage']}% missing values
        - {data_analysis['outlier_count']} outliers
        - {data_analysis['skewness']} skewness
        - {data_analysis['data_types']}

        What are the best preprocessing strategies?
        """

        # Retrieve relevant knowledge
        relevant_docs = self.vector_store.similarity_search(query, k=5)

        # Use LLM to synthesize recommendations
        recommendations = self.llm.invoke([
            ("system", "You are a data preprocessing expert."),
            ("user", f"Context: {relevant_docs}\n\nQuery: {query}")
        ])

        return recommendations
```

### Knowledge Base Content Structure
```
preprocessing_knowledge/
├── missing_data/
│   ├── numeric_imputation.md
│   ├── categorical_imputation.md
│   └── advanced_methods.md
├── outliers/
│   ├── detection_methods.md
│   ├── handling_strategies.md
│   └── domain_specific.md
├── encoding/
│   ├── onehot_encoding.md
│   ├── label_encoding.md
│   └── target_encoding.md
└── scaling/
    ├── standardization.md
    ├── normalization.md
    └── robust_scaling.md
```

---

## 4. Chat Interface for Each Agent

### Backend API

**New Endpoints**:
```python
POST /api/agent/chat
{
  "job_id": "abc123",
  "agent_id": "preprocessing",
  "message": "Why did you choose mean imputation?",
  "context": {}  # Current agent state
}

Response:
{
  "response": "I chose mean imputation because...",
  "references": ["step_2", "data_analysis"],
  "suggestions": ["Try median imputation if data is skewed"]
}
```

### Chat State Management
- Maintain conversation history per agent
- Agent can reference its own decisions and results
- Provide explanations with data-backed reasoning

### Frontend Component

```javascript
<AgentChat
  agentId="preprocessing"
  agentName="Preprocessing Agent"
  onSendMessage={handleChatMessage}
  conversationHistory={chatHistory}
/>
```

**Features**:
- Chat bubble UI
- Code snippets in responses
- Reference links to specific pipeline steps
- Export conversation

---

## 5. Backend Architecture Changes

### New Files Structure

```
backend/
├── agents/
│   ├── adaptive_agent_base.py      # NEW: Base class for adaptive agents
│   ├── chat_enabled_agent.py       # NEW: Mixin for chat capability
│   ├── sub_agents/                 # NEW: Specialized sub-agents
│   │   ├── outlier_analysis_agent.py
│   │   ├── feature_engineering_agent.py
│   │   └── data_balancing_agent.py
│   └── rag_agent.py                # NEW: RAG-powered recommendations
├── knowledge_base/                  # NEW: Preprocessing knowledge
│   ├── vector_store/
│   └── documents/
├── pipeline/
│   ├── dag_manager.py              # NEW: Dynamic DAG management
│   └── feedback_loop.py            # NEW: Agent learning mechanism
└── chat/
    ├── agent_chat_handler.py       # NEW: Chat endpoint logic
    └── conversation_manager.py     # NEW: Conversation state
```

### Modified Files
- `pipeline_orchestrator.py` - Support DAG updates
- `pipeline_state.py` - Add DAG and chat state
- `main.py` - New endpoints for chat and DAG

---

## 6. Implementation Phases

### Phase 1: DAG Visualization (1-2 hours)
1. Install React Flow
2. Create InteractiveDAG component
3. Update backend to return DAG structure
4. Test rendering with current pipeline

### Phase 2: Adaptive Agents (2-3 hours)
1. Create AdaptiveAgentBase class
2. Implement spawning logic in PreprocessingAgent
3. Create 2-3 specialized sub-agents
4. Update DAG manager to track dynamic agents
5. Test agent spawning

### Phase 3: RAG Agent (2-3 hours)
1. Create knowledge base documents
2. Set up vector store (Chroma)
3. Implement RAGAgent
4. Integrate with preprocessing pipeline
5. Test recommendations

### Phase 4: Chat Interface (2-3 hours)
1. Create chat API endpoints
2. Add ChatMixin to agents
3. Build AgentChat React component
4. Implement conversation state management
5. Test chat with each agent

### Phase 5: Integration & Testing (1-2 hours)
1. End-to-end testing
2. UI/UX refinements
3. Documentation updates
4. Performance optimization

**Total Estimated Time**: 8-13 hours

---

## 7. Key Technologies

**Frontend**:
- React Flow - DAG visualization
- WebSockets (optional) - Real-time updates

**Backend**:
- LangChain - RAG implementation
- Chroma/FAISS - Vector database
- FastAPI WebSockets - Real-time communication

**Storage**:
- JSON files - Conversation history
- SQLite (optional) - Pipeline states
- Vector DB - Knowledge base

---

## 8. Success Criteria

✅ **DAG Visualization**:
- Can see all agents and sub-agents in graph
- Nodes update in real-time
- Can interact with nodes (click, expand)

✅ **Adaptive Agents**:
- Agent spawns sub-agent when needed
- Sub-agent appears in DAG
- Results integrated back into pipeline

✅ **RAG Agent**:
- Provides relevant recommendations
- Cites knowledge base sources
- Improves preprocessing decisions

✅ **Chat Interface**:
- Can ask questions to any agent
- Agent provides contextual answers
- References its own decisions
- Conversation history maintained

---

## 9. Quick Start Implementation

Since this is comprehensive, let's start with a **Minimum Viable Enhancement** (MVE):

### MVE Scope:
1. ✅ Basic DAG visualization with React Flow
2. ✅ One adaptive sub-agent example (OutlierAnalysisAgent)
3. ✅ Simple RAG with pre-defined knowledge
4. ✅ Chat interface for preprocessing agent only

**This can be built in ~4-5 hours and demonstrates all core concepts.**

---

Would you like me to proceed with:
A) Full implementation (8-13 hours)
B) MVE implementation (4-5 hours)
C) Start with specific component (which one?)
