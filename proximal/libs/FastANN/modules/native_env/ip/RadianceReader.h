/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _RGBEREAD_H
#define _RGBEREAD_H

#include "SystemFile.h"
#include "ManagedPtr.h"

enum RGBEError
{
    RGBE_OK, RGBE_IO_ERROR, RGBE_OUT_OF_MEMORY, RGBE_CORRUPTED_DATA, RGBE_FORMAT_NOT_SUPPORTED,
};

#define RGBE_HEADER_GAMMA    0x01
#define RGBE_HEADER_EXPOSURE 0x02

#define RGBE_DATA_RGBA_FLOAT 0
#define RGBE_DATA_RGBE       1

class RGBEReader: public ManagedObject<RGBEReader>
{
public:
    ~RGBEReader(void);

    int getImageWidth(void)
    {
        return mHeader.width;
    }

    int getImageHeight(void)
    {
        return mHeader.height;
    }

    float getImageGamma(void)
    {
        return mHeader.gamma;
    }

    float getImageExposure(void)
    {
        return mHeader.exposure;
    }

    uint getImageFlags(void)
    {
        return mHeader.flags;
    }

    RGBEError readPixels(void *data, int dataType);

    static managed_ptr<RGBEReader> Open(const char *name);

private:
    RGBEReader(managed_ptr<System::File> const &fp);
    RGBEReader(RGBEReader const &instance);
    RGBEReader &operator=(RGBEReader const &instance);

    RGBEError readHeader(void);
    RGBEError readPixelsFloatRGB(float *data, int pixelsRead);
    RGBEError readPixelsRGBE(int *data, int pixelsRead);

    struct
    {
        int width, height, flags;
        float gamma, exposure;
    } mHeader;

    uchar *mBuffer;
    float mExpLUT[256];
    managed_ptr<System::File> mFile;
};

class RGBEWriter: public ManagedObject<RGBEWriter>
{
public:
    ~RGBEWriter(void);

    RGBEError writePixels(float *data, int numpixels);

    static managed_ptr<RGBEWriter> Open(const char *name, int width, int height);

private:
    RGBEWriter(managed_ptr<System::File> const &fp);
    RGBEWriter(RGBEWriter const &instance);
    RGBEWriter &operator=(RGBEWriter const &instance);

    RGBEError writeHeader(int width, int height);

    managed_ptr<System::File> mFile;
};

#endif 

