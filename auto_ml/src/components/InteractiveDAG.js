import React, { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './InteractiveDAG.css';

const nodeTypes = {};

function InteractiveDAG({ jobId, onNodeClick }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);

  // Fetch DAG structure from backend
  useEffect(() => {
    if (!jobId) return;

    const fetchDAG = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/pipeline/dag/${jobId}`);
        const data = await response.json();

        // Convert backend DAG format to React Flow format
        const flowNodes = data.nodes.map((node) => ({
          id: node.id,
          type: node.type === 'sub_agent' ? 'default' : 'default',
          data: {
            label: (
              <div className={`dag-node-content ${node.type}`}>
                <div className="node-label">{node.label}</div>
                <div className={`node-status ${node.status}`}>{node.status}</div>
                {node.type === 'sub_agent' && (
                  <div className="node-badge">SUB</div>
                )}
              </div>
            ),
            nodeData: node
          },
          position: node.position,
          className: `dag-node ${node.type} ${node.status}`,
          style: {
            background: getNodeColor(node.type, node.status),
            border: node.type === 'sub_agent' ? '2px solid #9333ea' : '2px solid #3b82f6',
            borderRadius: '8px',
            padding: '10px',
            width: 200,
          },
        }));

        const flowEdges = data.edges.map((edge, idx) => {
          const edgeColors = {
            sequential: '#3b82f6',
            spawned: '#9333ea',
            consults: '#f59e0b',
            informs: '#10b981',
            reports: '#8b5cf6'
          };

          const edgeType = edge.type || 'sequential';
          const color = edgeColors[edgeType] || '#3b82f6';

          return {
            id: `e${edge.from}-${edge.to}-${idx}`,
            source: edge.from,
            target: edge.to,
            type: edgeType === 'sequential' ? 'default' : 'smoothstep',
            animated: ['consults', 'spawned', 'informs'].includes(edgeType),
            label: edge.label || '',
            style: {
              stroke: color,
              strokeWidth: edgeType === 'sequential' ? 3 : 2,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: color,
            },
            labelStyle: {
              fontSize: 11,
              fill: color,
              fontWeight: 600,
            },
            labelBgStyle: {
              fill: 'white',
              fillOpacity: 0.8,
            },
          };
        });

        setNodes(flowNodes);
        setEdges(flowEdges);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch DAG:', error);
        setLoading(false);
      }
    };

    fetchDAG();
    // Poll for updates every 3 seconds
    const interval = setInterval(fetchDAG, 3000);

    return () => clearInterval(interval);
  }, [jobId, setNodes, setEdges]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const handleNodeClick = useCallback(
    (event, node) => {
      if (onNodeClick) {
        onNodeClick(node.data.nodeData);
      }
    },
    [onNodeClick]
  );

  if (loading) {
    return <div className="dag-loading">Loading pipeline DAG...</div>;
  }

  return (
    <div className="dag-container">
      <div className="dag-header">
        <h3>Interactive Pipeline DAG</h3>
        <div className="dag-legend">
          <div className="legend-item">
            <div className="legend-box main-agent"></div>
            <span>Main Stage</span>
          </div>
          <div className="legend-item">
            <div className="legend-box decision-agent"></div>
            <span>Decision Agent</span>
          </div>
          <div className="legend-item">
            <div className="legend-box sub-agent"></div>
            <span>Sub-Agent</span>
          </div>
          <div className="legend-item">
            <div className="legend-box edge-sequential"></div>
            <span>Sequential</span>
          </div>
          <div className="legend-item">
            <div className="legend-box edge-consults"></div>
            <span>Consults</span>
          </div>
          <div className="legend-item">
            <div className="legend-box edge-informs"></div>
            <span>Informs</span>
          </div>
        </div>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        fitView
        attributionPosition="bottom-left"
      >
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const nodeData = node.data.nodeData;
            return getNodeColor(nodeData?.type, nodeData?.status);
          }}
          nodeStrokeWidth={3}
        />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

function getNodeColor(type, status) {
  // Decision/RAG agent colors
  if (type === 'decision' || type === 'rag' || type === 'analysis') {
    if (status === 'completed') return '#f59e0b';
    if (status === 'running') return '#fbbf24';
    return '#fef3c7';
  }

  // Sub-agent colors
  if (type === 'sub_agent') {
    if (status === 'completed') return '#a855f7';
    if (status === 'running') return '#c084fc';
    return '#e9d5ff';
  }

  // Main agent colors
  if (status === 'completed') return '#10b981';
  if (status === 'running') return '#3b82f6';
  if (status === 'failed') return '#ef4444';
  return '#cbd5e1';
}

export default InteractiveDAG;
