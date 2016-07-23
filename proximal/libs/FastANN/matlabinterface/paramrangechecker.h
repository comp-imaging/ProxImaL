/*
 * paramrangechecker.h
 *
 *  Created on: Feb 5, 2014
 *      Author: msteinberger
 *  Copyright 2008 NVIDIA Corporation.  All Rights Reserved
 */

#ifndef PARAMRANGECHECKER_H_
#define PARAMRANGECHECKER_H_

#include <utils/Misc.cuh>


static void checkParameter(int minP, int maxP, int incstep, int mulstep, int& param, std::string const& paramname, std::ostream& message)
{
    bool paramok = false;
    for (int i = minP; i <= maxP; i = i*mulstep + incstep)
    {
        if (i == param)
        {
            paramok = true;
            break;
        }
        if (param < i)
        {
            message << "Unsupported " << paramname << " " << param << " setting " << paramname <<  " to " << i << "\n";
            param = i;
            paramok = true;
            break;
        }
    }
    if (!paramok)
    {
        message << "Unsupported " << paramname << " " << param << " setting " << paramname <<  " to " << maxP << "\n";
        param = maxP;
    }
}

template<int kTileSize, int kBlockSize, int kMaxTiles>
static TileStep getTilestep(int width, int height)
{
    int maxTilesX = 0, maxTilesY = 0, runs = 0x3FFFFFFF;

    for (int subX = 1; width / subX >= kTileSize; subX *= 2)
    {
        int tilesX = DIVUP(DIVUP(width,subX), kTileSize);
        int tilesY = min(kMaxTiles / tilesX, DIVUP(height, kTileSize));

        int runsX = DIVUP(width, tilesX*kTileSize);
        int runsY = DIVUP(height, tilesY*kTileSize);
        if (runsX * runsY < runs)
            maxTilesX = tilesX, maxTilesY = tilesY, runs = runsX * runsY;
    }

    if (maxTilesX == 0 || maxTilesY == 0)
    {
        printf("Warning: could not derive any tile stepping!?\n");
        maxTilesX = 4;
        maxTilesY = 4;
    }
    return TileStep(maxTilesX, maxTilesY);
}

#endif //PARAMRANGECHECKER_H_
