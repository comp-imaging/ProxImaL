#define BM3D_KEEP_DC 1
#define BM3D_KAISER 1
#define BM3D_SORT_DIM3 1


#include "gpuinterface.h"
#include "paramrangechecker.h"
#include <Core/FrameBufferHandle.h>
#include <structures/DataGroup.cuh>
#include <NonLocalMean.h>
#include <NonLocalMean.cuh>
#include <../tests/pipe/private/testTemplateIterator.h>

struct NLMParams
{
    ConstChannels in;
    Channels out;
    float noiseSigma;
    float outputGain;
    int width, height;
    NlmChannelInfo ci;

    std::vector<float> timings;
    float overalltime;
    float memusage;
    std::ostream & message;

    NLMParams(NlmChannelInfo ci, ConstChannels in,Channels out, int width, int height, float noiseSigma, float outputGain, std::ostream & message) :
            in(in), out(out), noiseSigma(noiseSigma), outputGain(outputGain), width(width), height(height), ci(ci), overalltime(-1.0f), memusage(0), message(message)
    {
    }
};


class NlmGPUCaller
{
public:
    template<class InputParameters, class TemplateParameters>
    static bool call(InputParameters& nlm_params, const TemplateParameterSelection & callSelection, int level)
    {
        // algo
        const int kAlgo = TempalteIteratorParameterGetter<TemplateParameters, 0>::value;
        const NlmDenoiseAlgorithm kAlgorithm = kAlgo == 0 ? NLM_DNA_BM3D :
                                               kAlgo == 1 ? NLM_DNA_BM3D_WIENER :
                                               kAlgo == 2 ? NLM_DNA_WEIGHTED_AVERAGE :
                                               NLM_DNA_AVERAGE;
        // blocksize
        const int kBlockSize= TempalteIteratorParameterGetter<TemplateParameters, 2>::value;
        // tilesize
        const int kTileSize= TempalteIteratorParameterGetter<TemplateParameters, 3>::value;
        // pixelstep
        const int kPixelStep = TempalteIteratorParameterGetter<TemplateParameters, 4>::value;
        // clustersize
        const int kClusterSize= TempalteIteratorParameterGetter<TemplateParameters, 5>::value;
        // candidatesize
        const int kCandidateSize = TempalteIteratorParameterGetter<TemplateParameters, 6>::value;


        int width = nlm_params.width, height = nlm_params.height;
        const int kMaxPixelsAtOnce = 1024 * 1024;
        const int kMaxTiles = DIVUP(kMaxPixelsAtOnce, kTileSize*kTileSize);

        TileStep tilestep = getTilestep<kTileSize,kBlockSize, kMaxTiles>(width, height);

        //nlm_params.message << kAlgorithm << " " << kBlockSize << " " << kTileSize << " " << kPixelStep << " " << kClusterSize << " " << kCandidateSize << " dims: " << width << " " << height << "\n";

        NonLocalMean<kAlgorithm, NLM_CLN_SINGLE_CLUSTER,
                kBlockSize, kTileSize, kPixelStep, kClusterSize, kCandidateSize, kMaxTiles, true> nlm;
        nlm_params.memusage = nlm.init(width, height, tilestep, false, nlm_params.ci);
        const EventRecord* timings = nlm.denoise(nlm_params.ci, nlm_params.out, nlm_params.in, nlm_params.noiseSigma, nlm_params.outputGain);

        nlm_params.overalltime = timings->get();
        nlm_params.timings = std::vector<float>(4,0.0f);
        for(int i = 0; i + 4 < timings->numRecords(); i += 5)
        {
            nlm_params.timings[0] += timings->get(i, i + 1);
            nlm_params.timings[1] += timings->get(i + 1, i + 2);
            nlm_params.timings[2] += timings->get(i + 2, i + 3);
            nlm_params.timings[3] += timings->get(i + 3, i + 4);
        }

        nlm.cleanUp();

        return true;
    }
};



bool runDenoiser(Config& config, float sigma, ConstChannels inputData, Channels outputData, float& memusage, float& exectime, std::ostream& message)
{
    int width = inputData[0].Width();
    int height = inputData[0].Height();

    if (config.Algorithm == NLM_DNA_DCT_SOFTTHRESHOLD)
    {
        if(config.BlockSize != 8)
            message << "SlidingDCT only supported with BlockSize 8, setting BlockSize to 8\n";
        NonLocalMean<NLM_DNA_DCT_SOFTTHRESHOLD, NLM_CLN_SINGLE_CLUSTER,
                        8, 25, 1, 32, 16, DIVUP(1024*1024, 25*25), false, false> nlm;
        memusage = nlm.init(width, height, TileStep(1024/25, 1024/25));

        const EventRecord* timings = nlm.denoise(NLM_CI_GREYSCALE, outputData, inputData, sigma, 1.0f);
        exectime = timings->get();
        nlm.cleanUp();
        return true;
    }


    const int kMinBlockSize = 8;  //8;
    const int kMaxBlockSize = 8; //8;
    checkParameter(kMinBlockSize, kMaxBlockSize, 0, 2, config.BlockSize, "BlockSize", message);
    if((config.Algorithm == NLM_DNA_BM3D || config.Algorithm == NLM_DNA_BM3D_WIENER) && config.BlockSize > 8)
    {
        message << "BM3D only supported with BlockSize 4 and 8, setting BlockSize to 8\n";
        config.BlockSize = 8;
    }

    const int kMinTileSize = 15; //15;
    const int kMaxTileSize = 15; //15;
    const int kTileSizeStepping = 4;
    checkParameter(kMinTileSize, kMaxTileSize, kTileSizeStepping, 1, config.TileSize, "TileSize", message);


    const int kMinClusterSize = 32; //16
    const int kMaxClusterSize = 32; //32
    const int kClusterSizeStepping = 16;
    checkParameter(kMinClusterSize, kMaxClusterSize, kClusterSizeStepping, 1, config.ClusterSize, "ClusterSize", message);

    const int kMinCandiateSize = 16; //8
    const int kMaxCandiateSize = 16; //16
    const int kCandiateSizeStepping = 8;
    checkParameter(kMinCandiateSize, kMaxCandiateSize, kCandiateSizeStepping, 1, config.NumCandidates, "NumCandidates", message);

    typedef
    LinearTemplateIterator<0,3, 1, // algo -
    LinearTemplateIterator<0,0, 1, // neighborhood - deactivated
    Pow2TemplateIterator<kMinBlockSize, kMaxBlockSize,   // blocksize 4,16,
    LinearTemplateIterator<kMinTileSize, kMaxTileSize, kTileSizeStepping, // tilesize 9,32, 3,
    LinearTemplateIterator<1,1,  1, // pixelstep
    LinearTemplateIterator<kMinClusterSize, kMaxClusterSize, kClusterSizeStepping, // clustersize 16,32, 4,
    LinearTemplateIterator<kMinCandiateSize, kMaxCandiateSize, kCandiateSizeStepping, // candidatesize 4,16, 2,
       NlmGPUCaller > > > > > > >  SettingsIterator;

    NLMParams params(config.Format, inputData, outputData, width, height, sigma, 1.0f, message);
    TemplateParameterSelection selection;
    switch(config.Algorithm)
    {
    case NLM_DNA_BM3D: selection.push_back(0); break;
    case NLM_DNA_BM3D_WIENER: selection.push_back(1); break;
    case NLM_DNA_WEIGHTED_AVERAGE: selection.push_back(2); break;
    case NLM_DNA_AVERAGE: selection.push_back(3); break;
    default:
        return false;
    }
    selection.push_back(0);
    selection.push_back(config.BlockSize);
    selection.push_back(config.TileSize);
    selection.push_back(1);
    selection.push_back(config.ClusterSize);
    selection.push_back(config.NumCandidates);

    if(!SettingsIterator :: call<NLMParams>(params,selection) )
        return false;

    memusage = params.memusage;
    exectime = params.overalltime;
    return true;
}
