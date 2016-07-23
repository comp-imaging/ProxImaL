/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <math.h>
#include <stdio.h>
#include "ip/RadianceReader.h"
#include "SystemCore.h"

#define RGBE_DATA_RED    0
#define RGBE_DATA_GREEN  1
#define RGBE_DATA_BLUE   2
#define RGBE_DATA_SIZE   3

#define STRING_BUFFER_SIZE   128
#define NUM_DELTA            0.0000000001f

// ========================================================================
// READER
// ========================================================================

RGBEReader::RGBEReader(managed_ptr<System::File> const &fp)
        : mHeader(), mBuffer(0), mFile(fp)
{
}

RGBEReader::~RGBEReader(void)
{
    if (mBuffer != 0)
    {
        delete[] mBuffer;
    }
}

RGBEError RGBEReader::readHeader(void)
{
    char buffer[STRING_BUFFER_SIZE];

    if (mBuffer != 0)
    {
        return RGBE_IO_ERROR;
    }

    // magic number check
    if (mFile->readLine(buffer, STRING_BUFFER_SIZE) == 0)
    {
        return RGBE_IO_ERROR;
    }

    if (buffer[0] != '#' || buffer[1] != '?')
    {
        return RGBE_CORRUPTED_DATA;
    }

    // read header tag data
    bool correctInput = false;
    mHeader.flags = 0;
    mHeader.exposure = 1.0f;

    for (;;)
    {
        if (mFile->readLine(buffer, STRING_BUFFER_SIZE) == 0)
        {
            return RGBE_IO_ERROR;
        }

        if (System::StringCompare(buffer, "\n"))
        {
            break;
        }

        if (System::StringCompare(buffer, "FORMAT=32-bit_rle_rgbe\n"))
        {
            correctInput = true;
        }
        else if (sscanf(buffer, "GAMMA=%f", &mHeader.gamma) == 1)
        {
            mHeader.flags |= RGBE_HEADER_GAMMA;
        }
        else if (sscanf(buffer, "EXPOSURE=%f", &mHeader.exposure) == 1)
        {
            mHeader.flags |= RGBE_HEADER_EXPOSURE;
        }
    }

    if (!correctInput)
    {
        return RGBE_FORMAT_NOT_SUPPORTED;
    }

    // read image dimensions
    if (mFile->readLine(buffer, STRING_BUFFER_SIZE) == 0)
    {
        return RGBE_IO_ERROR;
    }

    if (sscanf(buffer, "-Y %i +X %i", &mHeader.height, &mHeader.width) < 2)
    {
        return RGBE_CORRUPTED_DATA;
    }

    mBuffer = new uchar[mHeader.width << 2];

    return RGBE_OK;
}

static FORCE_INLINE void RGBEToFloatRGB(float *dst, unsigned char *rgbe, float *expLUT)
{
    const float f = expLUT[rgbe[3]];
    dst[0] = rgbe[2] * f + NUM_DELTA;
    dst[1] = rgbe[1] * f + NUM_DELTA;
    dst[2] = rgbe[0] * f + NUM_DELTA;
    dst[3] = 0.0f;
}

RGBEError RGBEReader::readPixelsFloatRGB(float *data, int pixelsRead)
{
    if (!mFile->readBytes(mBuffer + pixelsRead, (mHeader.width - pixelsRead) << 2))
    {
        return RGBE_IO_ERROR;
    }

    for (int j = pixelsRead; j < mHeader.width; j++)
    {
        RGBEToFloatRGB(data, mBuffer + (j << 2), mExpLUT);
        data += 4;
    }

    for (int i = 1; i < mHeader.height; i++)
    {
        if (!mFile->readBytes(mBuffer, mHeader.width << 2))
        {
            return RGBE_IO_ERROR;
        }

        for (int j = 0; j < mHeader.width; j++)
        {
            RGBEToFloatRGB(data, mBuffer + (j << 2), mExpLUT);
            data += 4;
        }
    }

    return RGBE_OK;
}

RGBEError RGBEReader::readPixelsRGBE(int *data, int pixelsRead)
{
    if (!mFile->readBytes(data, (mHeader.width * mHeader.height - pixelsRead) << 2))
    {
        return RGBE_IO_ERROR;
    }

    return RGBE_OK;
}

RGBEError RGBEReader::readPixels(void *data, int dataType)
{
    // init transformation LUT
    const float stonits = 179.0f / mHeader.exposure;

    mExpLUT[0] = 0.0f;
    for (int i = 1; i < 256; i++)
    {
        mExpLUT[i] = ldexpf(1.0f, i - 136) * stonits;
    }

    if (mHeader.width < 8 || mHeader.width > 0x7fff)
    {
        if (dataType == RGBE_DATA_RGBA_FLOAT)
        {
            return readPixelsFloatRGB((float *)data, 0);
        }
        else
        {
            return readPixelsRGBE((int *)data, 0);
        }
    }

    int width = mHeader.width;
    int height = mHeader.height;

    while (height > 0)
    {
        // read scanline info
        if (mFile->readBytes(mBuffer, 4) == 0)
        {
            return RGBE_IO_ERROR;
        }

        if (mBuffer[0] != 2 || mBuffer[1] != 2 || (mBuffer[2] & 0x80) != 0)
        {
            if (dataType == RGBE_DATA_RGBA_FLOAT)
            {
                RGBEToFloatRGB((float *)data, mBuffer, mExpLUT);
                return readPixelsFloatRGB((float *)data + 4, 1);
            }
            else
            {
                unsafe_pointer_cast<int>(data)[0] = unsafe_pointer_cast<int>(mBuffer)[0];
                return readPixelsRGBE((int *)data + 1, 1);
            }
        }

        // check scanline size
        if ((mBuffer[2] << 8 | mBuffer[3]) != width)
        {
            return RGBE_CORRUPTED_DATA;
        }

        // read 4 components to scanline buffer
        const int endIndex = width << 2;

        for (int i = 0; i < 4; i++)
        {
            int index = i;
            while (index < endIndex)
            {
                int count = mFile->readByte();
                int value = mFile->readByte();
                if (count < 0 || value < 0)
                {
                    return RGBE_IO_ERROR;
                }

                if (count > 128)
                {
                    // rle block
                    count -= 128;
                    if ((index >> 2) + count > width)
                    {
                        return RGBE_CORRUPTED_DATA;
                    }

                    for (int j = 0; j < count; j++)
                    {
                        mBuffer[index] = value;
                        index += 4;
                    }
                }
                else
                {
                    // not rle block
                    if (count == 0 || (index >> 2) + count > width)
                    {
                        return RGBE_CORRUPTED_DATA;
                    }

                    mBuffer[index] = value;
                    index += 4;
                    for (int j = 1; j < count; j++)
                    {
                        value = mFile->readByte();
                        if (value < 0)
                        {
                            return RGBE_IO_ERROR;
                        }
                        mBuffer[index] = value;
                        index += 4;
                    }
                }
            }
        }

        if (dataType == RGBE_DATA_RGBA_FLOAT)
        {
            for (int i = 0; i < width; i++)
            {
                RGBEToFloatRGB((float *)data, mBuffer + (i << 2), mExpLUT);
                data = (float *)data + 4;
            }
        }
        else
        {
            System::MemoryCopy(data, mBuffer, sizeof(int) * width);
            data = (int *)data + width;
        }

        height--;
    }

    return RGBE_OK;
}

managed_ptr<RGBEReader> RGBEReader::Open(const char *name)
{
    managed_ptr<System::File> fp(new System::File());

    if (!fp->open(name, System::File::FILE_READ))
    {
        LOG( "file not found!");
        return managed_ptr<RGBEReader>();
    }

    managed_ptr<RGBEReader> instance(new RGBEReader(fp));
    if (instance->readHeader() != RGBE_OK)
    {
        LOG( "corrupted header!");
        return managed_ptr<RGBEReader>();
    }

    return instance;
}

// ========================================================================
// WRITER
// ========================================================================

RGBEWriter::RGBEWriter(managed_ptr<System::File> const &fp)
        : mFile(fp)
{

}

RGBEWriter::~RGBEWriter(void)
{

}

FORCE_INLINE uint FloatRGBToRGBE(float red, float green, float blue)
{
    // XXX: this version works well for common cases (no support for denorms, negative numbers, inf/nans)
    float v = red;
    if (green > v)
    {
        v = green;
    }
    if (blue > v)
    {
        v = blue;
    }
    if (v < 1e-32f)
    {
        return 0;
    }
    else
    {
        union
        {
            int i;
            float f;
        } fi;

        fi.f = v;
        int e = fi.i >> 23;
        fi.i = (fi.i & 0x7fffff) | 0x43000000;
        v = fi.f / v;

        return ((e + 2) << 24) | (((int)(blue * v)) << 16) | (((int)(green * v)) << 8) | ((int)(red * v));
    }
}

/* default minimal header. modify if you want more information in header */
RGBEError RGBEWriter::writeHeader(int width, int height)
{
    char buffer[STRING_BUFFER_SIZE];
    sprintf(buffer, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n");
    mFile->writeBytes(buffer, System::StringLength(buffer));
    sprintf(buffer, "-Y %d +X %d\n", height, width);
    mFile->writeBytes(buffer, System::StringLength(buffer));

    return RGBE_OK;
}

/* simple write routine that does not use run length encoding */
RGBEError RGBEWriter::writePixels(float *data, int numPixels)
{
    uint *rgbe = new uint[numPixels];

    for (int i = 0; i < numPixels; i++)
    {
        rgbe[i] = FloatRGBToRGBE(data[RGBE_DATA_RED], data[RGBE_DATA_GREEN], data[RGBE_DATA_BLUE]);
        data += RGBE_DATA_SIZE;
    }

    mFile->writeBytes(rgbe, numPixels << 2);

    delete[] rgbe;

    return RGBE_OK;
}

managed_ptr<RGBEWriter> RGBEWriter::Open(const char *name, int width, int height)
{
    managed_ptr<System::File> fp(new System::File());

    if (!fp->open(name, System::File::FILE_WRITE))
    {
        return managed_ptr<RGBEWriter>();
    }

    managed_ptr<RGBEWriter> instance(new RGBEWriter(fp));
    if (instance->writeHeader(width, height) != RGBE_OK)
    {
        return managed_ptr<RGBEWriter>();
    }

    return instance;
}

