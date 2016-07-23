#include "gpuinterface_separate.h"
#include <structures/DataGroup.cuh>
#include <NonLocalMean.h>
#include <NonLocalMean.cuh>
#include "paramrangechecker.h"
#include <../tests/pipe/private/testTemplateIterator.h>


const int kMinBlockSize = 8; //4
const int kMaxBlockSize = 8; //16

const int kMinTileSize = 15; //11
const int kMaxTileSize = 15; //19
const int kTileSizeStepping = 4;

const int kMinClusterSize = 32; //16
const int kMaxClusterSize = 32; //32
const int kClusterSizeStepping = 16;

const int kMinCandidateSize = 16;  //8
const int kMaxCandidateSize = 16; //16
const int kCandidateSizeStepping = 8;

struct IndexingParams
{
    ConstChannels in;
    int width, height;
    NlmChannelInfo ci;

    std::vector<int2>* candidates;
    std::vector<int2>& clusteredPixCoords;
    std::vector<int>& clusterOffsets;
    std::vector<int>& clusterSizes;

    float overalltime;
    float memusage;
    std::ostream & message;

    IndexingParams(std::vector<int2>& clusteredPixCoords, std::vector<int>& clusterOffsets, std::vector<int>& clusterSizes, NlmChannelInfo ci, ConstChannels in, int width, int height, std::ostream & message) :
            in(in), width(width), height(height), ci(ci), candidates(0), clusteredPixCoords(clusteredPixCoords), clusterOffsets(clusterOffsets), clusterSizes(clusterSizes), overalltime(-1.0f), memusage(0), message(message)
    {
    }

    IndexingParams(std::vector<int2>& candidates, std::vector<int2>& clusteredPixCoords, std::vector<int>& clusterOffsets, std::vector<int>& clusterSizes, NlmChannelInfo ci, ConstChannels in, int width, int height, std::ostream & message) :
            in(in), width(width), height(height), ci(ci), candidates(&candidates), clusteredPixCoords(clusteredPixCoords), clusterOffsets(clusterOffsets), clusterSizes(clusterSizes), overalltime(-1.0f), memusage(0), message(message)
    {
    }
};

class IndexingGPUCaller
{
public:
    template<class InputParameters, class TemplateParameters>
    static bool call(InputParameters& nlm_params, const TemplateParameterSelection & callSelection, int level)
    {
        nv::FrameBufferHandle buildOnFB = nlm_params.in[0];
        if(nlm_params.ci == NLM_CI_RGB && nlm_params.in.numChannels() == 3)
            buildOnFB = nlm_params.in[1];


        // blocksize
        const int kBlockSize= TempalteIteratorParameterGetter<TemplateParameters, 0>::value;
        // tilesize
        const int kTileSize= TempalteIteratorParameterGetter<TemplateParameters, 1>::value;
        // clustersize
        const int kClusterSize= TempalteIteratorParameterGetter<TemplateParameters, 2>::value;

        //nlm_params.message << kBlockSize << " " << kTileSize << " " << kClusterSize << "\n";

        int width = nlm_params.width, height = nlm_params.height;
        const int kMaxPixelsAtOnce = 1024 * 1024;
        const int kMaxTiles = DIVUP(kMaxPixelsAtOnce, kTileSize*kTileSize);

        TileStep tilestep = getTilestep<kTileSize,kBlockSize, kMaxTiles>(width, height);

        typedef KTrees<kBlockSize, kTileSize, 1, kClusterSize, kClusterSize/2, false, false, kMaxTiles, ClusterorSubsampleSorting< 1 > > MTree;
        MTree trees;
        size_t numBlocks = kTileSize * kTileSize * tilestep.xStep * tilestep.yStep;
        nlm_params.memusage = trees.init(tilestep.xStep * tilestep.yStep, numBlocks);

        TextureBlock<kBlockSize>* mTextureBlockPool; /**< references to the input texture */
        int2* mClusterInfo; /**< references to the leaves of the tree */
        cudaMalloc(reinterpret_cast<void**>(&mTextureBlockPool), sizeof(TextureBlock<kBlockSize> ) * numBlocks);
        nlm_params.memusage += sizeof(TextureBlock<kBlockSize> ) * numBlocks;
        cudaMalloc(reinterpret_cast<void**>(&mClusterInfo), sizeof(int2) * numBlocks);
        nlm_params.memusage += sizeof(int2) * numBlocks;


        const int sTileX = 0, sTileY = 0;
        const int eTileX = DIVUP(width - ROUNDED_HALF(kBlockSize) + 1, kTileSize);
        const int eTileY = DIVUP(height - ROUNDED_HALF(kBlockSize) + 1, kTileSize);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        nlm_params.overalltime = 0;

        for (int tiley = sTileY; tiley < eTileY; tiley += tilestep.yStep)
            for (int tilex = sTileX; tilex < eTileX; tilex += tilestep.xStep)
            {
                int lowerTileX = tilex, lowerTileY = tiley;
                int upperTileX = min(tilex + tilestep.xStep, eTileX);
                int upperTileY = min(tiley + tilestep.yStep, eTileY);
                int xTiles = upperTileX - lowerTileX;
                int yTiles = upperTileY - lowerTileY;

                // set up blocks array
                cudaEventRecord(start);
                LinkBlocks<kBlockSize, kTileSize>(mTextureBlockPool, buildOnFB, lowerTileX, upperTileX, lowerTileY, upperTileY,
                        width, height);
                //run clustering
                int numClusters = trees.buildIndex(mTextureBlockPool, mClusterInfo, lowerTileX, upperTileX, lowerTileY,
                        upperTileY, width, height);
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                float ttime;
                cudaEventElapsedTime(&ttime, start, end);
                nlm_params.overalltime += ttime;

                //copy to cpu
                size_t startBlock = nlm_params.clusteredPixCoords.size();
                nlm_params.clusteredPixCoords.reserve(startBlock + xTiles * yTiles * kTileSize * kTileSize);
                std::vector<TextureBlock<kBlockSize> > tBlockRefs(xTiles * yTiles * kTileSize * kTileSize);
                cudaMemcpy(&tBlockRefs[0], mTextureBlockPool,
                        sizeof(TextureBlock<kBlockSize> ) * xTiles * yTiles * kTileSize * kTileSize,
                        cudaMemcpyDeviceToHost);
                for (int i = 0; i < xTiles * yTiles * kTileSize * kTileSize; ++i)
                    nlm_params.clusteredPixCoords.push_back(
                            make_int2(tBlockRefs[i].posX() - kBlockSize / 2, tBlockRefs[i].posY() - kBlockSize / 2));
                std::vector<int2> tClusterInfo(numClusters);
                cudaMemcpy(&tClusterInfo[0], mClusterInfo, sizeof(int2) * numClusters, cudaMemcpyDeviceToHost);
                nlm_params.clusterOffsets.reserve(nlm_params.clusterOffsets.size() + numClusters);
                nlm_params.clusterSizes.reserve(nlm_params.clusterSizes.size() + numClusters);
                for (int i = 0; i < numClusters; ++i)
                    nlm_params.clusterOffsets.push_back(startBlock + tClusterInfo[i].x),
                    nlm_params.clusterSizes.push_back(tClusterInfo[i].y);
            }

        trees.cleanUp();
        cudaFree(mTextureBlockPool);
        cudaFree(mClusterInfo);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        return true;
    }
};

class QueryGPUCaller
{
public:
    template<class InputParameters, class TemplateParameters>
    static bool call(InputParameters& nlm_params, const TemplateParameterSelection & callSelection, int level)
    {
        nv::FrameBufferHandle buildOnFB = nlm_params.in[0];
        if(nlm_params.ci == NLM_CI_RGB && nlm_params.in.numChannels() == 3)
            buildOnFB = nlm_params.in[1];

        if(nlm_params.candidates == 0)
            return false;

        std::vector<int2>& candidates(*nlm_params.candidates);

        // blocksize
        const int kBlockSize= TempalteIteratorParameterGetter<TemplateParameters, 0>::value;
        // tilesize
        const int kTileSize= TempalteIteratorParameterGetter<TemplateParameters, 1>::value;
        // clustersize
        const int kClusterSize= TempalteIteratorParameterGetter<TemplateParameters, 2>::value;
        // clustersize
        const int kCandidateSize= TempalteIteratorParameterGetter<TemplateParameters, 3>::value;

        // powTwo
        const int kPowTwo = TempalteIteratorParameterGetter<TemplateParameters, 4>::value;

        int width = nlm_params.width, height = nlm_params.height;
        const int kMaxPixelsAtOnce = 1024 * 1024;
        const int kMaxTiles = DIVUP(kMaxPixelsAtOnce, kTileSize*kTileSize);


        TileStep tilestep = getTilestep<kTileSize,kBlockSize, kMaxTiles>(width, height);

        typedef KTrees<kBlockSize, kTileSize, 1, kClusterSize, kClusterSize/2, false, kPowTwo, kMaxTiles, ClusterorSubsampleSorting< 1 > > MTree;
        MTree trees;
        size_t numBlocks = kTileSize * kTileSize * tilestep.xStep * tilestep.yStep;
        nlm_params.memusage = trees.init(tilestep.xStep * tilestep.yStep, numBlocks);

        TextureBlock<kBlockSize>* mTextureBlockPool; /**< references to the input texture */
        int2* mClusterInfo; /**< references to the leaves of the tree */
        cudaMalloc(reinterpret_cast<void**>(&mTextureBlockPool), sizeof(TextureBlock<kBlockSize> ) * numBlocks);
        nlm_params.memusage += sizeof(TextureBlock<kBlockSize> ) * numBlocks;
        cudaMalloc(reinterpret_cast<void**>(&mClusterInfo), sizeof(int2) * numBlocks);
        nlm_params.memusage += sizeof(int2) * numBlocks;


        const int sTileX = 0, sTileY = 0;
        const int eTileX = DIVUP(width - ROUNDED_HALF(kBlockSize) + 1, kTileSize);
        const int eTileY = DIVUP(height - ROUNDED_HALF(kBlockSize) + 1, kTileSize);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        nlm_params.overalltime = 0;

        size_t usedClusters = 0;
        size_t startBlock = 0;
        std::vector<int2> tClusterInfo;

        for (int tiley = sTileY; tiley < eTileY; tiley += tilestep.yStep)
            for (int tilex = sTileX; tilex < eTileX; tilex += tilestep.xStep)
            {
                int lowerTileX = tilex, lowerTileY = tiley;
                int upperTileX = min(tilex + tilestep.xStep, eTileX);
                int upperTileY = min(tiley + tilestep.yStep, eTileY);
                int xTiles = upperTileX - lowerTileX;
                int yTiles = upperTileY - lowerTileY;

                //copy input data to GPU
                std::vector<TextureBlock<kBlockSize> > tBlockRefs(xTiles * yTiles * kTileSize * kTileSize);
                for (int i = 0; i < xTiles * yTiles * kTileSize * kTileSize; ++i)
                {
                    if(usedClusters+i >= nlm_params.clusteredPixCoords.size())
                    {
                        nlm_params.message << "clusterdIndexData out of bounds: " << usedClusters+i << " >= " << nlm_params.clusteredPixCoords.size() << "\n";
                        continue;
                    }
                    tBlockRefs[i].setPosition(nlm_params.clusteredPixCoords[usedClusters+i].x + kBlockSize / 2,
                            nlm_params.clusteredPixCoords[usedClusters+i].y + kBlockSize / 2);
                }

                // run link
                LinkBlocks<kBlockSize, kTileSize>(mTextureBlockPool, buildOnFB, lowerTileX, upperTileX, lowerTileY, upperTileY,
                                       width, height);

                //replace link result
                cudaMemcpy(mTextureBlockPool, &tBlockRefs[0], sizeof(TextureBlock<kBlockSize> ) * xTiles * yTiles * kTileSize * kTileSize, cudaMemcpyHostToDevice);

                tClusterInfo.clear();
                int numClusters = 0;
                for(; startBlock + numClusters < nlm_params.clusterOffsets.size(); ++numClusters)
                {
                    int i = startBlock + numClusters;
                    int2 clusterinfo = make_int2(nlm_params.clusterOffsets[i] - usedClusters,
                            nlm_params.clusterSizes[i]);
                    if(clusterinfo.x + clusterinfo.y > xTiles * yTiles * kTileSize * kTileSize)
                    {
                        //nlm_params.message << "breaking with " << clusterinfo.x << " " << clusterinfo.y << " due to " << xTiles * yTiles * kTileSize * kTileSize << "\n";
                        break;
                    }
                    //nlm_params.message << "adding cluster " << clusterinfo.x << " " << clusterinfo.y << "\n";
                    tClusterInfo.push_back(clusterinfo);
                }
                cudaMemcpy(mClusterInfo, &tClusterInfo[0], sizeof(int2) * numClusters, cudaMemcpyHostToDevice);
                //nlm_params.message << "calling with " << numClusters << " clusters\n";

                cudaEventRecord(start);
                typename MTree::MCanditatesPack* mCandidateLists = trees.search(mTextureBlockPool, mClusterInfo, lowerTileX, upperTileX,
                                                lowerTileY, upperTileY, width, height, 0, numClusters);

                cudaEventRecord(end);
                cudaEventSynchronize(end);
                float ttime;
                cudaEventElapsedTime(&ttime, start, end);
                nlm_params.overalltime += ttime;

                if(mCandidateLists == 0)
                {
                    nlm_params.message << "candidates retreival failed\n";
                    return false;
                }

                //copy result to cpu

                std::vector<typename MTree::MCanditatesPack> tCandidatesInfo(xTiles * yTiles * kTileSize * kTileSize);

                cudaMemcpy(&tCandidatesInfo[0], mCandidateLists, sizeof(typename MTree::MCanditatesPack) * xTiles * yTiles * kTileSize * kTileSize, cudaMemcpyDeviceToHost);

                for(int i = 0; i < tClusterInfo.size(); ++i)
                {
                    for(int j = tClusterInfo[i].x; j < tClusterInfo[i].x+tClusterInfo[i].y; ++j)
                    {
                        if(j >= tBlockRefs.size())
                        {
                            nlm_params.message << "block access out of bounds: " << j << " >= " << tBlockRefs.size() << "\n";
                            continue;
                        }

                        int x = tBlockRefs[j].posX() - kBlockSize/2;
                        int y = tBlockRefs[j].posY() - kBlockSize/2;
                        int* tdata = reinterpret_cast<int*>(&tCandidatesInfo[j]);
                        //if(x == 0 && y == 0)
                        //nlm_params.message << "block " << x << " " << y << " has candidate data: " << tdata[0] << " " << tdata[1] << "\n";

                        typename MTree::MCanditatesPack::AccessIterator it = tCandidatesInfo[j].access();
                        int candidate;
                        for(int c = 0; (candidate = it.hGetNextCandidate()) > -1; )
                        {
                            if(candidate >= tBlockRefs.size())
                            {
                               nlm_params.message << "candidate id out of bounds: " << candidate << " >= " << tBlockRefs.size() << "\n";
                               continue;
                            }

                            int candX = tBlockRefs[candidate].posX() - kBlockSize/2;
                            int candY = tBlockRefs[candidate].posY() - kBlockSize/2;
                            //if(x == 0 && y == 0)
                            //    nlm_params.message << "block " << x << " " << y << " has candidate: " << candX << " " << candY << "\n";
                            candidates[(y*width+x)*kCandidateSize+c] = make_int2(candX, candY);

                            ++c;
                        }
                    }
                }

                startBlock += numClusters;
                usedClusters += xTiles * yTiles * kTileSize * kTileSize;
            }

        trees.cleanUp();
        cudaFree(mTextureBlockPool);
        cudaFree(mClusterInfo);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        return true;
    }
};



bool runIndexing(std::vector<int2>& clusteredPixCoords, std::vector<int>& clusterOffsets, std::vector<int>& clusterSizes, nv::FrameBufferHandle& in, int& TileSize, int& BlockSize, int& ClusterSize, float& memusage, float& runtime, std::ostream& message)
{

    int width = in.Width();
    int height = in.Height();

    checkParameter(kMinBlockSize, kMaxBlockSize, 0, 2, BlockSize, "BlockSize", message);
    checkParameter(kMinTileSize, kMaxTileSize, kTileSizeStepping, 1, TileSize, "TileSize", message);
    checkParameter(kMinClusterSize, kMaxClusterSize, kClusterSizeStepping, 1, ClusterSize, "ClusterSize", message);

    typedef Pow2TemplateIterator<kMinBlockSize, kMaxBlockSize,
    LinearTemplateIterator<kMinTileSize, kMaxTileSize, kTileSizeStepping,
    LinearTemplateIterator<kMinClusterSize, kMaxClusterSize, kClusterSizeStepping,
       IndexingGPUCaller > > > SettingsIterator;

    IndexingParams params(clusteredPixCoords, clusterOffsets,clusterSizes, NLM_CI_GREYSCALE, in, width, height, message);
    TemplateParameterSelection selection;

    selection.push_back(BlockSize);
    selection.push_back(TileSize);
    selection.push_back(ClusterSize);

    if(!SettingsIterator :: call<IndexingParams>(params,selection) )
        return false;

    memusage = params.memusage;
    runtime = params.overalltime;
    return true;
}



bool runQuery(std::vector<int2>& candidates, nv::FrameBufferHandle& in, std::vector<int2> & clusteredPixCoords, std::vector<int> & clusterOffsets, std::vector<int> & clusterSizes, int& TileSize, int& BlockSize, int& ClusterSize, int& CandidateSize, float& memusage, float& runtime, std::ostream& message, bool PowTwoCandidates)
{
    int width = in.Width();
    int height = in.Height();

    int inBlockSize = BlockSize, inTileSize = TileSize, inClusterSize = ClusterSize;
    checkParameter(kMinBlockSize, kMaxBlockSize, 0, 2, BlockSize, "BlockSize", std::cout);
    checkParameter(kMinTileSize, kMaxTileSize, kTileSizeStepping, 1, TileSize, "TileSize", std::cout);
    checkParameter(kMinClusterSize, kMaxClusterSize, kClusterSizeStepping, 1, ClusterSize, "ClusterSize", std::cout);
    checkParameter(kMinCandidateSize, kMaxCandidateSize, kCandidateSizeStepping, 1, CandidateSize, "CandidateSize", message);
    if(inBlockSize != BlockSize)
    {
        message << "ERROR: Unsupported BlockSize requested! Rebuild Index with BlockSize " << BlockSize << "\n";
        return false;
    }
    if(inTileSize != TileSize)
    {
        message << "ERROR: Unsupported TileSize requested! Rebuild Index with TileSize " << TileSize << "\n";
        return false;
    }
    if(inClusterSize != ClusterSize)
    {
        message << "ERROR: Unsupported ClusterSize requested! Rebuild Index with ClusterSize " << ClusterSize << "\n";
        return false;
    }

    candidates.clear();
    candidates.resize(width*height*CandidateSize, make_int2(-1,-1));

    typedef Pow2TemplateIterator<kMinBlockSize, kMaxBlockSize,
            LinearTemplateIterator<kMinTileSize, kMaxTileSize, kTileSizeStepping,
                    LinearTemplateIterator<kMinClusterSize, kMaxClusterSize, kClusterSizeStepping,
                            LinearTemplateIterator<kMinCandidateSize, kMaxCandidateSize, kCandidateSizeStepping,
                               LinearTemplateIterator<0, 1, 1,
                                    QueryGPUCaller> > > > > SettingsIterator;


    IndexingParams params(candidates, clusteredPixCoords, clusterOffsets, clusterSizes, NLM_CI_GREYSCALE, in, width, height, message);
    TemplateParameterSelection selection;

    selection.push_back(BlockSize);
    selection.push_back(TileSize);
    selection.push_back(ClusterSize);
    selection.push_back(CandidateSize);
    selection.push_back(PowTwoCandidates?1:0);

    if (!SettingsIterator::call<IndexingParams>(params, selection))
        return false;

    memusage = params.memusage;
    runtime = params.overalltime;
    return true;
}
