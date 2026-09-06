#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define NUM_STREAMS 4 // Number of CUDA streams used for parallel batches

// For each node in the current frontier examine all its neighbors.
__global__ void bfsProcessFrontier(
    const int frontierCount,
    const int neighborCount,
    const int *adjacencyBlock,
    const int *frontierNodes,
    int *dVisited,
    int *dNextFlags)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierCount)
        return;

    const int baseIdx = tid * neighborCount;
      // Traverse all neighbors of this frontier node
    for (int neighIdx = 0; neighIdx < neighborCount; ++neighIdx)
    {
        int neighbor = adjacencyBlock[baseIdx + neighIdx];
        int old = atomicCAS(&dVisited[neighbor], 0, 1);
        if (old == 0)
        {
            atomicExch(&dNextFlags[neighbor], 1);
        }
    }
}
// Compute L2 distance between the start node and each frontier node's vector.
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

    float sumSq = 0.0f;
    for (int dimIdx = 0; dimIdx < dim; ++dimIdx)
    {
        float diff = frontierVectors[tid * dim + dimIdx] - startVector[dimIdx];
        sumSq += diff * diff;
    }

    distances[tid] = sqrtf(sumSq);
}
// Reads graph from CSV file
bool readGraphCsv(const string &path, vector<int> &adjacencyFlat, int &nodeCount, int &neighborCount, string &graphBaseName)
{
    ifstream in(path);
    if (!in.is_open())
    {
        cerr << "Failed to open graph CSV: " << path << "\n";
        return false;
    }

    {
        size_t pos = path.find_last_of("/\\");
        string fname = (pos == string::npos) ? path : path.substr(pos + 1);
        size_t dot = fname.find_last_of('.');
        graphBaseName = (dot == string::npos) ? fname : fname.substr(0, dot);
    }

    string header;
    if (!getline(in, header))
    {
        cerr << "CSV empty: " << path << "\n";
        return false;
    }

    vector<string> headerCols;
    {
        stringstream ss(header);
        string tok;
        while (getline(ss, tok, ','))
            headerCols.push_back(tok);
    }
    if (headerCols.size() < 2)
    {
        cerr << "Invalid header in CSV\n";
        return false;
    }

    neighborCount = (int)headerCols.size() - 1;

    vector<vector<int>> rows;
    string line;
    while (getline(in, line))
    {
        if (line.empty())
            continue;
        stringstream ss(line);
        string tok;
        vector<string> cols;
        while (getline(ss, tok, ','))
            cols.push_back(tok);
        if (cols.size() != (size_t)neighborCount + 1)
        {
            cerr << "Warning: skipping malformed CSV line\n";
            continue;
        }
        vector<int> neigh(neighborCount);
        for (int k = 0; k < neighborCount; ++k)
            neigh[k] = stoi(cols[k + 1]);
        rows.push_back(move(neigh));
    }
    in.close();

    nodeCount = (int)rows.size();
    adjacencyFlat.resize(nodeCount * neighborCount);
    for (int rowIdx = 0; rowIdx < nodeCount; ++rowIdx)
        for (int colIdx = 0; colIdx < neighborCount; ++colIdx)
            adjacencyFlat[rowIdx * neighborCount + colIdx] = rows[rowIdx][colIdx];

    return true;
}

// Reads .fvecs file (standard SIFT vector format)
bool readFvecs(const string &path, vector<float> &vectorsFlat, int &nVecs, int &dim)
{
    ifstream in(path, ios::binary);
    if (!in.is_open())
    {
        cerr << "Warning: failed to open fvecs file: " << path << "\n";
        return false;
    }

    in.seekg(0, ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, ios::beg);

    int d;
    if (!in.read(reinterpret_cast<char *>(&d), sizeof(int)))
    {
        cerr << "Warning: failed reading fvecs header\n";
        in.close();
        return false;
    }
    dim = d;
    if (dim <= 0 || dim > 10000)
    {
        cerr << "Warning: suspicious dimension in fvecs: " << dim << "\n";
        in.close();
        return false;
    }

    size_t vecBytes = sizeof(int) + sizeof(float) * dim;
    nVecs = (int)(fileSize / vecBytes);
    vectorsFlat.resize((size_t)nVecs * dim);
    in.seekg(0, ios::beg);
    for (int vecIdx = 0; vecIdx < nVecs; ++vecIdx)
    {
        int dd;
        in.read(reinterpret_cast<char *>(&dd), sizeof(int));
        if (dd != dim)
        {
            cerr << "Warning: inconsistent dim in fvecs\n";
            in.close();
            return false;
        }
        in.read(reinterpret_cast<char *>(vectorsFlat.data() + (size_t)vecIdx * dim), sizeof(float) * dim);
    }
    in.close();
    return true;
}

// Structure to store BFS statistics per level
struct LevelInfo
{
    int level;
    int count;
    int farId;
    float farDist;
    int closeId;
    float closeDist;
};

// Writes all results to output file and console
void writeOutput(const string &graphBaseName,
                 const vector<LevelInfo> &results,
                 double totalMs)
{
    string outName = "output_" + graphBaseName + "_task_3.txt";
    ofstream outf(outName);
    if (!outf.is_open())
    {
        cerr << "Warning: could not open output file " << outName << "\n";
    }

    for (const auto &li : results)
    {
        char buf[512];
        snprintf(buf, sizeof(buf), "%d,%d,%d,%.6f,%d,%.6f",
                 li.level, li.count,
                 (li.farId >= 0 ? li.farId : -1),
                 li.farDist,
                 (li.closeId >= 0 ? li.closeId : -1),
                 li.closeDist);

        cout << buf << "\n";
        if (outf.is_open())
            outf << buf << "\n";
    }

    string timeMsg = "Total BFS discovery time = " + to_string(totalMs) + " ms";
    cout << timeMsg << "\n";
    if (outf.is_open())
        outf << timeMsg << "\n";

    if (outf.is_open())
        outf.close();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: ./bfs <absolute_path_to_graph.csv> <absolute_path_to_sift_base.fvecs> <start_node>\n";
        return 1;
    }

    string graphPath = argv[1];
    string fvecsPath = argv[2];
    int startNode = atoi(argv[3]);
    vector<int> adjacencyFlat;
    int nodeCount = 0, neighborCount = 0;
    string graphBaseName;
    // Load graph CSV
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
    // Load vector embeddings for all nodes
    vector<float> vectorsFlat;
    int nVecs = 0, dim = 0;
    bool haveVectors = readFvecs(fvecsPath, vectorsFlat, nVecs, dim);
    if (!haveVectors || nVecs < nodeCount)
    {
        cerr << "Error: Need valid vectors for all nodes.\n";
        return 1;
    }

    vector<int> hVisited(nodeCount, 0);
    vector<int> hFrontierNodes;
    vector<int> hNextFrontierNodes;
    hFrontierNodes.push_back(startNode);
    hVisited[startNode] = 1;
    vector<LevelInfo> results;

    // Create CUDA streams for batched execution
    cudaStream_t streams[NUM_STREAMS];
    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
    {
        cudaStreamCreate(&streams[streamId]);
    }

    int *dVisited = nullptr;
    float *dStartVector = nullptr;

    cudaMalloc(&dVisited, sizeof(int) * nodeCount);
    cudaMalloc(&dStartVector, sizeof(float) * dim);

    cudaMemcpy(dVisited, hVisited.data(), sizeof(int) * nodeCount, cudaMemcpyHostToDevice);
    cudaMemcpy(dStartVector, vectorsFlat.data() + startNode * dim, sizeof(float) * dim, cudaMemcpyHostToDevice);

    auto t0 = chrono::steady_clock::now();
    int level = 0;

    while (!hFrontierNodes.empty())
    {
        int frontierCount = (int)hFrontierNodes.size();
        // Divide frontier into batches for multiple streams
        int batchSize = (frontierCount + NUM_STREAMS - 1) / NUM_STREAMS;

        int *hPinnedAdjacency[NUM_STREAMS] = {nullptr};
        int *hPinnedFrontier[NUM_STREAMS] = {nullptr};
        int *hPinnedNextFlags[NUM_STREAMS] = {nullptr};
        float *hPinnedVectors[NUM_STREAMS] = {nullptr};
        float *hPinnedDistances[NUM_STREAMS] = {nullptr};

        int *dAdjacencyBlock[NUM_STREAMS] = {nullptr};
        int *dFrontierNodes[NUM_STREAMS] = {nullptr};
        int *dNextFlags[NUM_STREAMS] = {nullptr};
        float *dFrontierVectors[NUM_STREAMS] = {nullptr};
        float *dDistances[NUM_STREAMS] = {nullptr};
        // Build a CUDA Graph containing all operations
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaGraphCreate(&graph, 0);

        vector<cudaGraphNode_t> memsetNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> h2dAdjNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> h2dFrontierNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> h2dVectorNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> bfsKernelNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> distKernelNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> d2hFlagsNodes(NUM_STREAMS);
        vector<cudaGraphNode_t> d2hDistNodes(NUM_STREAMS);
        // Create graph nodes for each stream batch
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            int startIdx = streamId * batchSize;
            int endIdx = min(startIdx + batchSize, frontierCount);
            int currentBatch = endIdx - startIdx;

            if (currentBatch <= 0)
                break;
            // Allocate pinned host memory
            cudaMallocHost(&hPinnedAdjacency[streamId], sizeof(int) * currentBatch * neighborCount);
            cudaMallocHost(&hPinnedFrontier[streamId], sizeof(int) * currentBatch);
            cudaMallocHost(&hPinnedNextFlags[streamId], sizeof(int) * nodeCount);
            cudaMallocHost(&hPinnedVectors[streamId], sizeof(float) * currentBatch * dim);
            cudaMallocHost(&hPinnedDistances[streamId], sizeof(float) * currentBatch);
            // Allocate device memory
            cudaMalloc(&dAdjacencyBlock[streamId], sizeof(int) * currentBatch * neighborCount);
            cudaMalloc(&dFrontierNodes[streamId], sizeof(int) * currentBatch);
            cudaMalloc(&dNextFlags[streamId], sizeof(int) * nodeCount);
            cudaMalloc(&dFrontierVectors[streamId], sizeof(float) * currentBatch * dim);
            cudaMalloc(&dDistances[streamId], sizeof(float) * currentBatch);
            // Fill pinned host memory with correct batch data
            for (int localIdx = 0; localIdx < currentBatch; ++localIdx)
            {
                int nodeIdxInFrontier = startIdx + localIdx;
                int node = hFrontierNodes[nodeIdxInFrontier];

                memcpy(hPinnedAdjacency[streamId] + localIdx * neighborCount,
                       adjacencyFlat.data() + node * neighborCount,
                       sizeof(int) * neighborCount);

                hPinnedFrontier[streamId][localIdx] = node;

                memcpy(hPinnedVectors[streamId] + localIdx * dim,
                       vectorsFlat.data() + node * dim,
                       sizeof(float) * dim);
            }
            // Add memset node for next flags
            cudaMemsetParams memsetParams = {};
            memsetParams.dst = dNextFlags[streamId];
            memsetParams.value = 0;
            memsetParams.pitch = 0;
            memsetParams.elementSize = sizeof(int);
            memsetParams.width = nodeCount;
            memsetParams.height = 1;

            cudaGraphAddMemsetNode(&memsetNodes[streamId], graph, nullptr, 0, &memsetParams);
            // Add memcpy nodes (adjacency, frontier, vectors)
            cudaMemcpy3DParms h2dAdjParams = {};
            h2dAdjParams.srcPtr = make_cudaPitchedPtr(hPinnedAdjacency[streamId],
                                                      currentBatch * neighborCount * sizeof(int),
                                                      currentBatch * neighborCount, 1);
            h2dAdjParams.dstPtr = make_cudaPitchedPtr(dAdjacencyBlock[streamId],
                                                      currentBatch * neighborCount * sizeof(int),
                                                      currentBatch * neighborCount, 1);
            h2dAdjParams.extent = make_cudaExtent(currentBatch * neighborCount * sizeof(int), 1, 1);
            h2dAdjParams.kind = cudaMemcpyHostToDevice;

            cudaGraphAddMemcpyNode(&h2dAdjNodes[streamId], graph, &memsetNodes[streamId], 1, &h2dAdjParams);
             // H2D frontier
            cudaMemcpy3DParms h2dFrontierParams = {};
            h2dFrontierParams.srcPtr = make_cudaPitchedPtr(hPinnedFrontier[streamId],
                                                           currentBatch * sizeof(int),
                                                           currentBatch, 1);
            h2dFrontierParams.dstPtr = make_cudaPitchedPtr(dFrontierNodes[streamId],
                                                           currentBatch * sizeof(int),
                                                           currentBatch, 1);
            h2dFrontierParams.extent = make_cudaExtent(currentBatch * sizeof(int), 1, 1);
            h2dFrontierParams.kind = cudaMemcpyHostToDevice;

            cudaGraphAddMemcpyNode(&h2dFrontierNodes[streamId], graph, &memsetNodes[streamId], 1, &h2dFrontierParams);
            // H2D vector embeddings
            cudaMemcpy3DParms h2dVectorParams = {};
            h2dVectorParams.srcPtr = make_cudaPitchedPtr(hPinnedVectors[streamId],
                                                         currentBatch * dim * sizeof(float),
                                                         currentBatch * dim, 1);
            h2dVectorParams.dstPtr = make_cudaPitchedPtr(dFrontierVectors[streamId],
                                                         currentBatch * dim * sizeof(float),
                                                         currentBatch * dim, 1);
            h2dVectorParams.extent = make_cudaExtent(currentBatch * dim * sizeof(float), 1, 1);
            h2dVectorParams.kind = cudaMemcpyHostToDevice;

            cudaGraphAddMemcpyNode(&h2dVectorNodes[streamId], graph, &memsetNodes[streamId], 1, &h2dVectorParams);
                // BFS kernel node
            int threads = 256;
            int blocks = (currentBatch + threads - 1) / threads;
                
            cudaKernelNodeParams bfsKernelParams = {};
            void *bfsArgs[] = {&currentBatch, &neighborCount, &dAdjacencyBlock[streamId],
                               &dFrontierNodes[streamId], &dVisited, &dNextFlags[streamId]};
            bfsKernelParams.func = (void *)bfsProcessFrontier;
            bfsKernelParams.gridDim = dim3(blocks, 1, 1);
            bfsKernelParams.blockDim = dim3(threads, 1, 1);
            bfsKernelParams.sharedMemBytes = 0;
            bfsKernelParams.kernelParams = bfsArgs;
            bfsKernelParams.extra = nullptr;

            cudaGraphNode_t bfsDeps[] = {h2dAdjNodes[streamId], h2dFrontierNodes[streamId]};
            cudaGraphAddKernelNode(&bfsKernelNodes[streamId], graph, bfsDeps, 2, &bfsKernelParams);
                // Distance kernel node
            cudaKernelNodeParams distKernelParams = {};
            void *distArgs[] = {&currentBatch, &dFrontierNodes[streamId], &dStartVector,
                                &dFrontierVectors[streamId], &dim, &dDistances[streamId]};
            distKernelParams.func = (void *)computeDistances;
            distKernelParams.gridDim = dim3(blocks, 1, 1);
            distKernelParams.blockDim = dim3(threads, 1, 1);
            distKernelParams.sharedMemBytes = 0;
            distKernelParams.kernelParams = distArgs;
            distKernelParams.extra = nullptr;

            cudaGraphNode_t distDeps[] = {h2dFrontierNodes[streamId], h2dVectorNodes[streamId]};
            cudaGraphAddKernelNode(&distKernelNodes[streamId], graph, distDeps, 2, &distKernelParams);
             // D2H next flags (after BFS kernel)
            cudaMemcpy3DParms d2hFlagsParams = {};
            d2hFlagsParams.srcPtr = make_cudaPitchedPtr(dNextFlags[streamId],
                                                        nodeCount * sizeof(int),
                                                        nodeCount, 1);
            d2hFlagsParams.dstPtr = make_cudaPitchedPtr(hPinnedNextFlags[streamId],
                                                        nodeCount * sizeof(int),
                                                        nodeCount, 1);
            d2hFlagsParams.extent = make_cudaExtent(nodeCount * sizeof(int), 1, 1);
            d2hFlagsParams.kind = cudaMemcpyDeviceToHost;

            cudaGraphAddMemcpyNode(&d2hFlagsNodes[streamId], graph, &bfsKernelNodes[streamId], 1, &d2hFlagsParams);
             // D2H distances (after distance kernel)
            cudaMemcpy3DParms d2hDistParams = {};
            d2hDistParams.srcPtr = make_cudaPitchedPtr(dDistances[streamId],
                                                       currentBatch * sizeof(float),
                                                       currentBatch, 1);
            d2hDistParams.dstPtr = make_cudaPitchedPtr(hPinnedDistances[streamId],
                                                       currentBatch * sizeof(float),
                                                       currentBatch, 1);
            d2hDistParams.extent = make_cudaExtent(currentBatch * sizeof(float), 1, 1);
            d2hDistParams.kind = cudaMemcpyDeviceToHost;

            cudaGraphAddMemcpyNode(&d2hDistNodes[streamId], graph, &distKernelNodes[streamId], 1, &d2hDistParams);
        }
           // Launch the CUDA graph and wait for completion
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
        // Gather all distances back into one array
        vector<float> hAllDistances(frontierCount);
        int offsetIdx = 0;
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            int startIdx = streamId * batchSize;
            int endIdx = min(startIdx + batchSize, frontierCount);
            int currentBatch = endIdx - startIdx;

            if (currentBatch <= 0)
                break;

            memcpy(hAllDistances.data() + offsetIdx,
                   hPinnedDistances[streamId],
                   sizeof(float) * currentBatch);
            offsetIdx += currentBatch;
        }
         // Merge next frontier flags from all streams
        vector<int> hNextFlags(nodeCount, 0);
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            if (hPinnedNextFlags[streamId] == nullptr)
                continue;

            for (int nid = 0; nid < nodeCount; ++nid)
            {
                if (hPinnedNextFlags[streamId][nid])
                    hNextFlags[nid] = 1;
            }
        }
         // Find farthest and closest nodes in this level
        int farId = -1, closeId = -1;
        float farDist = -1.0f;
        float closeDist = numeric_limits<float>::infinity();

        for (int idx = 0; idx < frontierCount; ++idx)
        {
            int node = hFrontierNodes[idx];
            float dist = hAllDistances[idx];

            if (dist > farDist)
            {
                farDist = dist;
                farId = node;
            }
            if (dist < closeDist)
            {
                closeDist = dist;
                closeId = node;
            }
        }

        results.push_back({level, frontierCount, farId, farDist, closeId, closeDist});

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);

        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            cudaFreeHost(hPinnedAdjacency[streamId]);
            cudaFreeHost(hPinnedFrontier[streamId]);
            cudaFreeHost(hPinnedNextFlags[streamId]);
            cudaFreeHost(hPinnedVectors[streamId]);
            cudaFreeHost(hPinnedDistances[streamId]);

            cudaFree(dAdjacencyBlock[streamId]);
            cudaFree(dFrontierNodes[streamId]);
            cudaFree(dNextFlags[streamId]);
            cudaFree(dFrontierVectors[streamId]);
            cudaFree(dDistances[streamId]);
        }

        hNextFrontierNodes.clear();
        for (int nid = 0; nid < nodeCount; ++nid)
            if (hNextFlags[nid])
                hNextFrontierNodes.push_back(nid);

        hFrontierNodes.swap(hNextFrontierNodes);
        ++level;
    }

    auto t1 = chrono::steady_clock::now();
    double totalMs = chrono::duration<double, milli>(t1 - t0).count();
     // Write results to output
    writeOutput(graphBaseName, results, totalMs);

    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
    {
        cudaStreamDestroy(streams[streamId]);
    }

    cudaFree(dVisited);
    cudaFree(dStartVector);
    cudaDeviceReset();

    return 0;
}