#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// This kernel takes the current BFS frontier and explores all neighbors.
// If a neighbor is unvisited, we mark it visited and flag it for the next frontier.
__global__ void bfsProcessFrontier(
    const int frontierCount,
    const int neighborCount,
    const int *adjacencyBlock,
    const int *frontierNodes,
    int *visited,
    int *nextFlags)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierCount)
        return;

    int baseIdx = tid * neighborCount;

    for (int i = 0; i < neighborCount; ++i)
    {
        int neighbor = adjacencyBlock[baseIdx + i];

        int old = atomicCAS(&visited[neighbor], 0, 1);
        if (old == 0)
        {
            // This is the first time the neighbor is discovered.
            // Mark it so it becomes part of the next frontier.
            atomicExch(&nextFlags[neighbor], 1);
        }
    }
}


// Kernel for computing distances between the BFS start vector
// and the vectors of the nodes inside the current frontier.
__global__ void computeDistances(
    const int frontierCount,
    const int *frontierNodes,
    const float *startVector,
    const float *frontierVectors,
    const int dim,
    float *distances)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierCount)
        return;

    float sum = 0.0f;

    for (int d = 0; d < dim; ++d)
    {
        float diff = frontierVectors[tid * dim + d] - startVector[d];
        sum += diff * diff;
    }

    distances[tid] = sqrtf(sum);
}

// Reads the graph CSV file and flattens adjacency lists.
bool readGraphCsv(const string &path, vector<int> &adjacencyFlat, int &nodeCount, int &neighborCount, string &graphBaseName)
{
    ifstream file(path);
    if (!file.is_open())
    {
        cerr << "Failed to open graph CSV: " << path << "\n";
        return false;
    }

    {
        size_t pos = path.find_last_of("/\\");
        string fileName = (pos == string::npos) ? path : path.substr(pos + 1);
        size_t dot = fileName.find_last_of('.');
        graphBaseName = (dot == string::npos) ? fileName : fileName.substr(0, dot);
    }

    string header;
    if (!getline(file, header))
    {
        cerr << "CSV empty: " << path << "\n";
        return false;
    }

    vector<string> headerCols;
    {
        stringstream ss(header);
        string token;
        while (getline(ss, token, ','))
            headerCols.push_back(token);
    }

    if (headerCols.size() < 2)
    {
        cerr << "Invalid CSV header\n";
        return false;
    }

    neighborCount = headerCols.size() - 1;

    vector<vector<int>> rowData;
    string line;

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        vector<string> cols;
        string token;

        while (getline(ss, token, ','))
            cols.push_back(token);

        if (cols.size() != (size_t)neighborCount + 1)
        {
            cerr << "Warning: skipping malformed CSV row\n";
            continue;
        }

        vector<int> neighbors(neighborCount);
        for (int i = 0; i < neighborCount; ++i)
            neighbors[i] = stoi(cols[i + 1]);

        rowData.push_back(move(neighbors));
    }

    file.close();

    nodeCount = rowData.size();
    adjacencyFlat.resize(nodeCount * neighborCount);

    for (int i = 0; i < nodeCount; ++i)
        for (int j = 0; j < neighborCount; ++j)
            adjacencyFlat[i * neighborCount + j] = rowData[i][j];

    return true;
}

// Reads .fvecs file and loads vectors into memory.
bool readFvecs(const string &path, vector<float> &vectorsFlat, int &vecCount, int &dim)
{
    ifstream file(path, ios::binary);
    if (!file.is_open())
    {
        cerr << "Warning: failed to open fvecs file: " << path << "\n";
        return false;
    }

    file.seekg(0, ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, ios::beg);

    int readDim;
    if (!file.read(reinterpret_cast<char *>(&readDim), sizeof(int)))
    {
        cerr << "Warning: failed reading fvecs header\n";
        file.close();
        return false;
    }

    dim = readDim;

    if (dim <= 0 || dim > 10000)
    {
        cerr << "Warning: suspicious dimension in fvecs: " << dim << "\n";
        file.close();
        return false;
    }

    size_t vecBytes = sizeof(int) + sizeof(float) * dim;
    vecCount = fileSize / vecBytes;

    vectorsFlat.resize((size_t)vecCount * dim);

    file.seekg(0, ios::beg);

    for (int i = 0; i < vecCount; ++i)
    {
        int dTemp;
        file.read(reinterpret_cast<char *>(&dTemp), sizeof(int));

        if (dTemp != dim)
        {
            cerr << "Warning: inconsistent dim in fvecs\n";
            file.close();
            return false;
        }

        file.read(reinterpret_cast<char *>(vectorsFlat.data() + (size_t)i * dim), sizeof(float) * dim);
    }

    file.close();
    return true;
}


// Struct storing BFS information for each level
struct LevelInfo
{
    int level;
    int count;
    int farNodeId;
    float farDistance;
    int closeNodeId;
    float closeDistance;
};

// Writes BFS results to both console and output file.
void writeOutput(
    const string &graphBaseName,
    const vector<LevelInfo> &results,
    double totalMs)
{
    string outputName = "output_" + graphBaseName + "_task_1.txt";
    ofstream out(outputName);

    if (!out.is_open())
        cerr << "Warning: could not open output file " << outputName << "\n";

    for (const auto &info : results)
    {
        char buffer[512];
        snprintf(
            buffer,
            sizeof(buffer),
            "%d,%d,%d,%.6f,%d,%.6f",
            info.level,
            info.count,
            (info.farNodeId >= 0 ? info.farNodeId : -1),
            info.farDistance,
            (info.closeNodeId >= 0 ? info.closeNodeId : -1),
            info.closeDistance);

        cout << buffer << "\n";
        if (out.is_open())
            out << buffer << "\n";
    }

    string timeMsg = "Total BFS discovery time = " + to_string(totalMs) + " ms";
    cout << timeMsg << "\n";

    if (out.is_open())
    {
        out << timeMsg << "\n";
        out.close();
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: ./bfs <path_to_graph.csv> <path_to_sift_base.fvecs> <start_node>\n";
        return 1;
    }

    string graphPath = argv[1];
    string fvecsPath = argv[2];
    int startNode = atoi(argv[3]);

    vector<int> adjacencyFlat;
    int nodeCount = 0, neighborCount = 0;
    string graphBaseName;
       // Load CSV graph
    if (!readGraphCsv(graphPath, adjacencyFlat, nodeCount, neighborCount, graphBaseName))
    {
        cerr << "Error reading graph CSV. Exiting.\n";
        return 1;
    }

    if (startNode < 0 || startNode >= nodeCount)
    {
        cerr << "Invalid start node index.\n";
        return 1;
    }

    vector<float> vectorsFlat;
    int vectorCount = 0, dim = 0;
    // Load SIFT vectors
    bool hasVectors = readFvecs(fvecsPath, vectorsFlat, vectorCount, dim);

    if (!hasVectors || vectorCount < nodeCount)
    {
        cerr << "Error: Need valid vectors for all nodes.\n";
        return 1;
    }
    // Host-side visited tracking and frontier lists
    vector<int> visitedHost(nodeCount, 0);
    vector<int> frontierNodesHost;
    vector<int> nextFrontierNodesHost;
     // Start BFS from the requested node
    frontierNodesHost.push_back(startNode);
    visitedHost[startNode] = 1;

    vector<LevelInfo> results;

    int *dVisited = nullptr;
    int *dNextFlags = nullptr;
    int *dAdjacencyBlock = nullptr;
    int *dFrontierNodes = nullptr;

    float *dStartVector = nullptr;
    float *dFrontierVectors = nullptr;
    float *dDistances = nullptr;

    cudaMalloc(&dVisited, sizeof(int) * nodeCount);
    cudaMalloc(&dNextFlags, sizeof(int) * nodeCount);
    cudaMalloc(&dStartVector, sizeof(float) * dim);
    cudaMalloc(&dDistances, sizeof(float) * nodeCount);

    cudaMemcpy(dVisited, visitedHost.data(), sizeof(int) * nodeCount, cudaMemcpyHostToDevice);
    cudaMemset(dNextFlags, 0, sizeof(int) * nodeCount);

    cudaMemcpy(
        dStartVector,
        vectorsFlat.data() + startNode * dim,
        sizeof(float) * dim,
        cudaMemcpyHostToDevice);

    auto beginTime = chrono::steady_clock::now();

    int level = 0;

    while (!frontierNodesHost.empty())
    {
        int frontierCount = frontierNodesHost.size();
        // Build a contiguous adjacency block for frontier nodes
        vector<int> adjacencyBlock(frontierCount * neighborCount);

        for (int i = 0; i < frontierCount; ++i)
        {
            int node = frontierNodesHost[i];
            int baseIdx = node * neighborCount;

            for (int j = 0; j < neighborCount; ++j)
                adjacencyBlock[i * neighborCount + j] = adjacencyFlat[baseIdx + j];
        }
        // Prepare frontier vectors to compute distances
        vector<float> frontierVectors(frontierCount * dim);

        for (int i = 0; i < frontierCount; ++i)
        {
            int node = frontierNodesHost[i];
            memcpy(
                frontierVectors.data() + i * dim,
                vectorsFlat.data() + node * dim,
                sizeof(float) * dim);
        }

        cudaMemset(dNextFlags, 0, sizeof(int) * nodeCount);

        cudaFree(dAdjacencyBlock);
        cudaFree(dFrontierNodes);
        cudaFree(dFrontierVectors);

        cudaMalloc(&dAdjacencyBlock, sizeof(int) * frontierCount * neighborCount);
        cudaMalloc(&dFrontierNodes, sizeof(int) * frontierCount);
        cudaMalloc(&dFrontierVectors, sizeof(float) * frontierCount * dim);

        cudaMemcpy(dAdjacencyBlock, adjacencyBlock.data(),
                   sizeof(int) * frontierCount * neighborCount, cudaMemcpyHostToDevice);
        cudaMemcpy(dFrontierNodes, frontierNodesHost.data(),
                   sizeof(int) * frontierCount, cudaMemcpyHostToDevice);
        cudaMemcpy(dFrontierVectors, frontierVectors.data(),
                   sizeof(float) * frontierCount * dim, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (frontierCount + threads - 1) / threads;
        // Launch BFS expansion kernel
        bfsProcessFrontier<<<blocks, threads>>>(
            frontierCount, neighborCount, dAdjacencyBlock, dFrontierNodes,
            dVisited, dNextFlags);
        cudaDeviceSynchronize();
        // Launch vector distance computation kernel
        computeDistances<<<blocks, threads>>>(
            frontierCount, dFrontierNodes, dStartVector,
            dFrontierVectors, dim, dDistances);
        cudaDeviceSynchronize();

        vector<int> nextFlagsHost(nodeCount);
        vector<float> distancesHost(frontierCount);

        cudaMemcpy(nextFlagsHost.data(), dNextFlags,
                   sizeof(int) * nodeCount, cudaMemcpyDeviceToHost);

        cudaMemcpy(distancesHost.data(), dDistances,
                   sizeof(float) * frontierCount, cudaMemcpyDeviceToHost);
        // Compute farthest and closest nodes in this frontier
        int farNode = -1, closeNode = -1;
        float farDist = -1.0f;
        float closeDist = numeric_limits<float>::infinity();

        for (int i = 0; i < frontierCount; ++i)
        {
            int node = frontierNodesHost[i];
            float d = distancesHost[i];

            if (d > farDist)
            {
                farDist = d;
                farNode = node;
            }
            if (d < closeDist)
            {
                closeDist = d;
                closeNode = node;
            }
        }

        results.push_back({level, frontierCount, farNode, farDist, closeNode, closeDist});
        // Build next frontier from flagged nodes
        nextFrontierNodesHost.clear();
        for (int i = 0; i < nodeCount; ++i)
            if (nextFlagsHost[i])
                nextFrontierNodesHost.push_back(i);

        frontierNodesHost.swap(nextFrontierNodesHost);
        ++level;
    }

    auto endTime = chrono::steady_clock::now();
    double totalMs = chrono::duration<double, milli>(endTime - beginTime).count();

    writeOutput(graphBaseName, results, totalMs);

    cudaFree(dAdjacencyBlock);
    cudaFree(dFrontierNodes);
    cudaFree(dVisited);
    cudaFree(dNextFlags);
    cudaFree(dStartVector);
    cudaFree(dFrontierVectors);
    cudaFree(dDistances);

    cudaDeviceReset();
    return 0;
}
