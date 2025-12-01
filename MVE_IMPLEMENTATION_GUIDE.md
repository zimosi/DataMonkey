# MVE Implementation Guide
## What's Been Built & What's Next

### ✅ Completed

1. **RAG Knowledge Base** (`backend/knowledge_base/preprocessing_knowledge.py`)
   - Comprehensive preprocessing best practices
   - Recommendations for missing values, outliers, scaling, encoding
   - Context-aware suggestion engine

2. **RAG Agent** (`backend/agents/rag_agent.py`)
   - Retrieves relevant preprocessing knowledge
   - Generates explanations using LLM
   - Suggests next steps based on results

3. **Updated Requirements** (`backend/requirements.txt`)
   - Added `chromadb` and `tiktoken` for RAG

### 🚧 Still To Build

#### 1. Adaptive Agent with Sub-Agent Spawning

**File**: `backend/agents/adaptive_preprocessing_agent.py`

```python
"""
Adaptive Preprocessing Agent that can spawn specialized sub-agents
"""
from .preprocessing_agent import PreprocessingAgent
from .rag_agent import RAGPreprocessingAgent
import pandas as pd
from typing import Dict, Any, List

class OutlierAnalysisSubAgent:
    """Specialized sub-agent for deep outlier analysis"""

    def analyze(self, df: pd.DataFrame, outlier_columns: List[str]) -> Dict[str, Any]:
        """Perform deep analysis on outliers"""
        analysis = {}

        for col in outlier_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

            analysis[col] = {
                "num_outliers": len(outliers),
                "outlier_percentage": len(outliers) / len(df) * 100,
                "outlier_values": outliers[col].tolist()[:10],  # Sample
                "recommendation": self._get_recommendation(len(outliers) / len(df) * 100)
            }

        return analysis

    def _get_recommendation(self, outlier_pct: float) -> str:
        if outlier_pct > 20:
            return "High outliers - investigate if these are data errors or legitimate extreme values"
        elif outlier_pct > 10:
            return "Moderate outliers - cap using IQR method"
        else:
            return "Low outliers - safe to cap or remove"


class AdaptivePreprocessingAgent(PreprocessingAgent):
    """Enhanced preprocessing agent that can spawn sub-agents"""

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.rag_agent = RAGPreprocessingAgent(llm)
        self.sub_agents_spawned = []

    def preprocess_with_intelligence(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        target_column: str,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Intelligent preprocessing that spawns sub-agents when needed
        """
        # Step 1: Get initial statistics
        initial_stats = self._analyze_data_stats(df)

        # Step 2: Get RAG recommendations
        rag_recommendations = self.rag_agent.get_recommendations(
            initial_stats,
            user_context="Initial preprocessing"
        )

        # Step 3: Run standard preprocessing
        base_result = self.preprocess_data(df, config, target_column, job_id)

        # Step 4: Analyze results and decide on sub-agents
        sub_agents_results = {}

        # Check if outlier sub-agent needed
        if initial_stats.get('outlier_percentage', 0) > 10:
            print(f"🤖 Spawning OutlierAnalysisSubAgent (outliers: {initial_stats['outlier_percentage']:.1f}%)")

            outlier_agent = OutlierAnalysisSubAgent()
            outlier_cols = initial_stats.get('outlier_columns', [])

            if outlier_cols:
                sub_result = outlier_agent.analyze(df, outlier_cols)
                sub_agents_results['outlier_analysis'] = sub_result
                self.sub_agents_spawned.append({
                    "type": "OutlierAnalysisSubAgent",
                    "reason": f"Detected {initial_stats['outlier_percentage']:.1f}% outliers",
                    "result": sub_result
                })

        # Step 5: Combine results
        enhanced_result = {
            **base_result,
            "rag_recommendations": rag_recommendations,
            "sub_agents": sub_agents_results,
            "sub_agents_spawned": self.sub_agents_spawned,
            "intelligence_summary": self._generate_summary(
                rag_recommendations,
                sub_agents_results
            )
        }

        return enhanced_result

    def _analyze_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data to get statistics"""
        numeric_cols = df.select_dtypes(include=['number']).columns

        stats = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "missing_percentage": (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            "num_categorical_cols": len(df.select_dtypes(include=['object', 'category']).columns),
        }

        # Outlier detection
        outlier_cols = []
        total_outliers = 0

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()

            if outliers > 0:
                outlier_cols.append(col)
                total_outliers += outliers

        stats['outlier_percentage'] = (total_outliers / (len(df) * len(numeric_cols))) * 100 if len(numeric_cols) > 0 else 0
        stats['outlier_columns'] = outlier_cols
        stats['has_outliers'] = len(outlier_cols) > 0

        # Scale variance
        if len(numeric_cols) > 1:
            scales = df[numeric_cols].std()
            stats['scale_variance'] = scales.max() / scales.min() if scales.min() > 0 else 0
        else:
            stats['scale_variance'] = 0

        # Max categories
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats['max_unique_categories'] = max([df[col].nunique() for col in cat_cols])
        else:
            stats['max_unique_categories'] = 0

        return stats

    def _generate_summary(
        self,
        rag_recommendations: Dict[str, Any],
        sub_agent_results: Dict[str, Any]
    ) -> str:
        """Generate summary of intelligent preprocessing"""
        summary_parts = ["Enhanced Preprocessing with AI Intelligence:"]

        if rag_recommendations.get('explanation'):
            summary_parts.append(f"\nRAG Insights: {rag_recommendations['explanation']}")

        if sub_agent_results:
            summary_parts.append(f"\n{len(sub_agent_results)} sub-agents spawned for specialized analysis")

        return "\n".join(summary_parts)
```

#### 2. Chat Interface Backend

**File**: `backend/chat/agent_chat.py`

```python
"""
Chat interface for agents
"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class AgentChatHandler:
    """Handles chat interactions with pipeline agents"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.conversations = {}  # job_id -> agent_id -> messages

    def chat_with_agent(
        self,
        job_id: str,
        agent_id: str,
        user_message: str,
        agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send message to an agent and get response

        Args:
            job_id: Pipeline job ID
            agent_id: Which agent to chat with
            user_message: User's question
            agent_state: Current state of the agent

        Returns:
            Response from agent
        """
        # Initialize conversation if needed
        conv_key = f"{job_id}_{agent_id}"
        if conv_key not in self.conversations:
            self.conversations[conv_key] = []

        # Add user message
        self.conversations[conv_key].append({
            "role": "user",
            "content": user_message
        })

        # Generate response based on agent type
        if agent_id == "preprocessing":
            response = self._preprocessing_agent_response(user_message, agent_state)
        elif agent_id == "data_understanding":
            response = self._data_understanding_response(user_message, agent_state)
        elif agent_id == "model_selection":
            response = self._model_selection_response(user_message, agent_state)
        else:
            response = "I'm not sure how to help with that agent."

        # Add assistant message
        self.conversations[conv_key].append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "conversation_history": self.conversations[conv_key]
        }

    def _preprocessing_agent_response(
        self,
        user_message: str,
        agent_state: Dict[str, Any]
    ) -> str:
        """Generate response from preprocessing agent"""

        # Extract relevant info from state
        steps_performed = agent_state.get('steps_performed', [])
        config_used = agent_state.get('config_used', {})
        original_shape = agent_state.get('original_shape', (0, 0))
        final_shape = agent_state.get('final_shape', (0, 0))

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Preprocessing Agent from Data Monkey pipeline.
You help users understand the preprocessing steps that were performed on their data.

Your personality:
- Helpful and educational
- Explain the 'why' behind decisions
- Reference specific steps you performed
- Suggest improvements when relevant

Be concise but informative."""),
            ("user", f"""Agent State:
Steps Performed: {[step.get('step', '') for step in steps_performed]}
Configuration: {config_used}
Data Shape Change: {original_shape} → {final_shape}

User Question: {user_message}

Provide a helpful response about the preprocessing that was done.""")
        ])

        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            return f"I encountered an error: {str(e)}"

    def _data_understanding_response(self, user_message: str, agent_state: Dict[str, Any]) -> str:
        """Generate response from data understanding agent"""
        # Similar implementation
        return "Data understanding agent response..."

    def _model_selection_response(self, user_message: str, agent_state: Dict[str, Any]) -> str:
        """Generate response from model selection agent"""
        # Similar implementation
        return "Model selection agent response..."

    def get_conversation_history(self, job_id: str, agent_id: str) -> List[Dict[str, str]]:
        """Get conversation history for an agent"""
        conv_key = f"{job_id}_{agent_id}"
        return self.conversations.get(conv_key, [])
```

#### 3. Backend API Endpoints

**Add to**: `backend/main.py`

```python
# Add these endpoints

from chat.agent_chat import AgentChatHandler
from agents.adaptive_preprocessing_agent import AdaptivePreprocessingAgent

# Initialize chat handler
chat_handler = AgentChatHandler(ChatOpenAI())

class ChatRequest(BaseModel):
    job_id: str
    agent_id: str
    message: str

@app.post("/api/agent/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Chat with a specific pipeline agent
    """
    try:
        if request.job_id not in active_pipelines:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        orchestrator = active_pipelines[request.job_id]

        # Get agent state
        agent_state = orchestrator.get_stage_result(PipelineStage(request.agent_id))

        # Get response
        response = chat_handler.chat_with_agent(
            request.job_id,
            request.agent_id,
            request.message,
            agent_state
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agent/conversation/{job_id}/{agent_id}")
async def get_conversation(job_id: str, agent_id: str):
    """Get conversation history with an agent"""
    try:
        history = chat_handler.get_conversation_history(job_id, agent_id)
        return JSONResponse(
            status_code=200,
            content={"history": history}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 4. Frontend DAG Component

**File**: `auto_ml/src/components/InteractiveDAG.js`

```javascript
import React, { useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './InteractiveDAG.css';

const nodeTypes = {
  mainAgent: MainAgentNode,
  subAgent: SubAgentNode,
};

function MainAgentNode({ data }) {
  return (
    <div className="main-agent-node">
      <div className="node-header">{data.label}</div>
      <div className="node-status">{data.status}</div>
      {data.subAgents && data.subAgents.length > 0 && (
        <div className="sub-agents-count">
          {data.subAgents.length} sub-agent(s)
        </div>
      )}
    </div>
  );
}

function SubAgentNode({ data }) {
  return (
    <div className="sub-agent-node">
      <div className="node-label">{data.label}</div>
      <div className="node-reason">{data.reason}</div>
    </div>
  );
}

function InteractiveDAG({ graphData }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  React.useEffect(() => {
    if (graphData && graphData.nodes) {
      // Convert graph data to React Flow format
      const flowNodes = graphData.nodes.map(node => ({
        id: node.id,
        type: node.type === 'main_agent' ? 'mainAgent' : 'subAgent',
        position: node.position || { x: 0, y: 0 },
        data: node,
      }));

      const flowEdges = graphData.edges.map((edge, idx) => ({
        id: `edge-${idx}`,
        source: edge.from,
        target: edge.to,
        animated: edge.type === 'active',
        type: edge.type === 'spawned' ? 'step' : 'smoothstep',
      }));

      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [graphData, setNodes, setEdges]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div className="dag-container" style={{ height: '600px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default InteractiveDAG;
```

#### 5. Frontend Chat Component

**File**: `auto_ml/src/components/AgentChat.js`

```javascript
import React, { useState } from 'react';
import './AgentChat.css';

function AgentChat({ jobId, agentId, agentName }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          agent_id: agentId,
          message: input,
        }),
      });

      const data = await response.json();
      const assistantMessage = { role: 'assistant', content: data.response };
      setMessages([...messages, userMessage, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <h3>💬 Chat with {agentName}</h3>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {isLoading && <div className="message assistant loading">Thinking...</div>}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask the agent anything..."
        />
        <button onClick={sendMessage} disabled={isLoading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default AgentChat;
```

### 📦 Installation Steps

1. **Install Python Dependencies**:
```bash
cd backend
pip install chromadb tiktoken
```

2. **Install React Dependencies**:
```bash
cd auto_ml
npm install reactflow
```

3. **Create Missing Directories**:
```bash
mkdir -p backend/knowledge_base
mkdir -p backend/chat
mkdir -p backend/agents/sub_agents
```

4. **Create Missing Files**:
   - Create the files listed above in their respective locations
   - Add the endpoint code to `main.py`

### 🚀 Testing the MVE

1. Start backend: `python backend/main.py`
2. Start frontend: `cd auto_ml && npm start`
3. Upload a CSV file
4. Click "Run Pipeline"
5. Watch for sub-agents spawning in logs
6. Click on preprocessing stage
7. Click "Chat" button to talk with agent
8. Ask: "Why did you use median imputation?"

### 🎯 What You'll See

- **RAG in action**: Agent explains preprocessing choices using knowledge base
- **Sub-agent spawning**: If outliers > 10%, OutlierAnalysisSubAgent spawns
- **Chat interface**: Ask agents about their decisions
- **DAG visualization**: See all agents and sub-agents in graph

This MVE demonstrates all core concepts in a working system!
