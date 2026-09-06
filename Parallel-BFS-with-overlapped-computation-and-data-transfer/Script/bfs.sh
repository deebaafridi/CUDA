#!/bin/bash

# Wrapper script to execute all three BFS tasks sequentially
# Usage: ./bfs.sh <path_to_graph.csv> <path_to_sift_base.fvecs> <start_node>

if [ "$#" -ne 3 ]; then
    echo "Usage: ./bfs.sh <absolute_path_to_graph.csv> <absolute_path_to_sift_base.fvecs> <start_node>"
    exit 1
fi

GRAPH_PATH="$1"
FVECS_PATH="$2"
START_NODE="$3"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# Execute Task 1: Baseline
echo "----------------------------------------"
echo "Task 1: Baseline BFS"
echo "----------------------------------------"
"$SCRIPT_DIR/bfs_baseline" "$GRAPH_PATH" "$FVECS_PATH" "$START_NODE"
if [ $? -ne 0 ]; then
    echo "Error: Task 1 execution failed"
    exit 1
fi
echo ""

# Execute Task 2: Stream-optimized
echo "----------------------------------------"
echo "Task 2: Stream-optimized BFS"
echo "----------------------------------------"
"$SCRIPT_DIR/bfs_stream" "$GRAPH_PATH" "$FVECS_PATH" "$START_NODE"
if [ $? -ne 0 ]; then
    echo "Error: Task 2 execution failed"
    exit 1
fi
echo ""

# Execute Task 3: CUDA Graph
echo "----------------------------------------"
echo "Task 3: CUDA Graph BFS"
echo "----------------------------------------"
"$SCRIPT_DIR/bfs_graph" "$GRAPH_PATH" "$FVECS_PATH" "$START_NODE"
if [ $? -ne 0 ]; then
    echo "Error: Task 3 execution failed"
    exit 1
fi
echo ""

