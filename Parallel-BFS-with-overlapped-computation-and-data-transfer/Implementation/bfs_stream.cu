#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define NUM_STREAMS 4    // Number of CUDA streams used for parallel batches

// Each thread handles one node from the frontier and checks all its neighbors.
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

    for (int neighIdx = 0; neighIdx < neighborCount; ++neighIdx)
    {
        int neighbor = adjacencyBlock[baseIdx + neighIdx];
        int old = atomicCAS(&dVisited[neighbor], 0, 1);
        if (old == 0)
        {
            // If this thread was the first to visit the neighbor mark it as a next-frontier candidate
            atomicExch(&dNextFlags[neighbor], 1);
        }
    }
}
//Compute Euclidean distance between frontier nodes and start node
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
// Reads a CSV graph: node_id,n0,n1,n2..
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

// Read fvecs file 
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

// struct to Collects BFS level summary
struct LevelInfo
{
    int level;
    int count;
    int farId;
    float farDist;
    int closeId;
    float closeDist;
};

// Writes output in file as well as console
void writeOutput(const string &graphBaseName,
                 const vector<LevelInfo> &results,
                 double totalMs)
{
    string outName = "output_" + graphBaseName + "_task_2.txt";
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
    // Load SIFT vectors
    vector<float> vectorsFlat;
    int nVecs = 0, dim = 0;
    bool haveVectors = readFvecs(fvecsPath, vectorsFlat, nVecs, dim);
    if (!haveVectors || nVecs < nodeCount)
    {
        cerr << "Error: Need valid vectors for all nodes.\n";
        return 1;
    }
    // Host-side visited tracking and frontier lists
    vector<int> hVisited(nodeCount, 0);
    vector<int> hFrontierNodes;
    vector<int> hNextFrontierNodes;
    hFrontierNodes.push_back(startNode);
    hVisited[startNode] = 1;
    vector<LevelInfo> results;

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
    {
        cudaStreamCreate(&streams[streamId]);
    }

    // Allocate persistent device memory (shared across all levels)
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

        // Process each stream's batch
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            int startIdx = streamId * batchSize;
            int endIdx = min(startIdx + batchSize, frontierCount);
            int currentBatch = endIdx - startIdx;

            if (currentBatch <= 0)
                break;

            cudaMallocHost(&hPinnedAdjacency[streamId], sizeof(int) * currentBatch * neighborCount);
            cudaMallocHost(&hPinnedFrontier[streamId], sizeof(int) * currentBatch);
            cudaMallocHost(&hPinnedNextFlags[streamId], sizeof(int) * nodeCount);
            cudaMallocHost(&hPinnedVectors[streamId], sizeof(float) * currentBatch * dim);
            cudaMallocHost(&hPinnedDistances[streamId], sizeof(float) * currentBatch);

            cudaMalloc(&dAdjacencyBlock[streamId], sizeof(int) * currentBatch * neighborCount);
            cudaMalloc(&dFrontierNodes[streamId], sizeof(int) * currentBatch);
            cudaMalloc(&dNextFlags[streamId], sizeof(int) * nodeCount);
            cudaMalloc(&dFrontierVectors[streamId], sizeof(float) * currentBatch * dim);
            cudaMalloc(&dDistances[streamId], sizeof(float) * currentBatch);
            // Fill pinned host buffers
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

            // Async copies
            cudaMemsetAsync(dNextFlags[streamId], 0, sizeof(int) * nodeCount, streams[streamId]);

            cudaMemcpyAsync(dAdjacencyBlock[streamId], hPinnedAdjacency[streamId],
                            sizeof(int) * currentBatch * neighborCount,
                            cudaMemcpyHostToDevice, streams[streamId]);
            cudaMemcpyAsync(dFrontierNodes[streamId], hPinnedFrontier[streamId],
                            sizeof(int) * currentBatch,
                            cudaMemcpyHostToDevice, streams[streamId]);
            cudaMemcpyAsync(dFrontierVectors[streamId], hPinnedVectors[streamId],
                            sizeof(float) * currentBatch * dim,
                            cudaMemcpyHostToDevice, streams[streamId]);

            int threads = 256;
            int blocks = (currentBatch + threads - 1) / threads;

                // Launch BFS kernel
            bfsProcessFrontier<<<blocks, threads, 0, streams[streamId]>>>(
                currentBatch, neighborCount, dAdjacencyBlock[streamId], dFrontierNodes[streamId],
                dVisited, dNextFlags[streamId]);
            // Launch distance kernel
            computeDistances<<<blocks, threads, 0, streams[streamId]>>>(
                currentBatch, dFrontierNodes[streamId], dStartVector,
                dFrontierVectors[streamId], dim, dDistances[streamId]);

            cudaMemcpyAsync(hPinnedNextFlags[streamId], dNextFlags[streamId],
                            sizeof(int) * nodeCount,
                            cudaMemcpyDeviceToHost, streams[streamId]);
            cudaMemcpyAsync(hPinnedDistances[streamId], dDistances[streamId],
                            sizeof(float) * currentBatch,
                            cudaMemcpyDeviceToHost, streams[streamId]);
        }
        // Wait for all streams to finish
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId)
        {
            cudaStreamSynchronize(streams[streamId]);
        }
        // Gather distance results from all streams
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
        // Merge next-frontier flags from all streams
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
        // Compute farthest/closest nodes for this level
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
        // Construct next frontier
        hNextFrontierNodes.clear();
        for (int nid = 0; nid < nodeCount; ++nid)
            if (hNextFlags[nid])
                hNextFrontierNodes.push_back(nid);

        hFrontierNodes.swap(hNextFrontierNodes);
        ++level;
    }

    auto t1 = chrono::steady_clock::now();
    double totalMs = chrono::duration<double, milli>(t1 - t0).count();

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