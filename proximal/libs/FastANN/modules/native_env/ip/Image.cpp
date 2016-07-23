/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#include "ip/ImagePixelUInt8x4.h"
#include "ip/Image.h"
#include "SystemFile.h"

#pragma pack(push, 1)

typedef struct S_BITMAP_HEADER
{
    ushort fileType;
    uint fileSize;
    ushort reserved[2];
    uint bitmapPtr;
} BITMAP_HEADER;

typedef struct
{
    uint headerSize;
    uint width;
    uint height;
    ushort bitplanesNum;
    ushort bpp;
    uint compression;
    uint size;
    uint pixelPerMeterX;
    uint pixelPerMeterY;
    uint colorsNum;
    uint importantColorsNum;
} BMPV3_INFO_HEADER;

typedef struct
{
    BMPV3_INFO_HEADER v3;
    uint colorSpace;
    uint colorSpaceData[16];
} BMPV5_INFO_HEADER;

#pragma pack(pop)

static bool StringEndsWith(const char *str, const char *pattern)
{
    int lstr = System::StringLength(str);
    int pstr = System::StringLength(pattern);

    if (lstr < pstr || lstr == 0 || pstr == 0)
    {
        return false;
    }

    int istart = lstr - pstr;
    for (int i = istart; i < lstr; i++)
    {
        if (str[i] != pattern[i - istart])
        {
            return false;
        }
    }

    return true;
}

template<> bool Image<IPF_uint8x4>::loadImage(const char *name)
{
    System::File f;
    if (!f.open(name, System::File::FILE_READ))
    {
        LOG_DEBUG( LOG_IO, "error opening %s!\n", name);
        return false;
    }

    if (StringEndsWith(name, ".bmp"))
    {
        BITMAP_HEADER head;
        BMPV3_INFO_HEADER info;

        f.readBytes(&head, sizeof(BITMAP_HEADER));
        f.readBytes(&info, sizeof(BMPV3_INFO_HEADER));

        if (info.bpp != 24 && info.bpp != 32)
        {
            LOG_DEBUG( LOG_IO, "unsupported bitmap type!");
            return false;
        }

        f.seek(head.bitmapPtr, System::File::SEEK_FROM_START);

        set(info.width, info.height);

        int scanSize = (mWidth * (info.bpp >> 3) + 3) & ~0x3;
        uchar *rdata = new uchar[scanSize];
        for (int y = mHeight - 1; y >= 0; y--)
        {
            f.readBytes(rdata, scanSize);
            // shuffle rgb
            int index = y * mWidth;
            if (info.bpp == 24)
            {
                for (uint i = 0; i < mWidth; i++)
                {
                    mData[index + i] = ImagePixelUInt8x4(rdata[i * 3 + 2], rdata[i * 3 + 1], rdata[i * 3], 255);
                }
            }
            else
            {
                for (uint i = 0; i < mWidth; i++)
                {
                    mData[index + i] = ImagePixelUInt8x4(rdata[i * 4 + 2], rdata[i * 4 + 1], rdata[i * 4],
                                                         rdata[i * 4 + 3]);
                }
            }
        }

        delete[] rdata;
    }
    else if (StringEndsWith(name, ".ppm"))
    {
        char temp[256];

        f.readLine(temp, 256); // read header
        if (System::StringCompare(temp, "P6") == 0)
        {
            f.readLine(temp, 256); // skip comment line
            int width, height;
            f.readLine(temp, 256);
            sscanf(temp, "%i %i\n", &width, &height);
            f.readLine(temp, 256); // always equal to "255" for P6

            // read data
            set(width, height);

            uchar *rdata = new uchar[width * 3];
            int index = 0;
            for (int y = 0; y < height; y++)
            {
                f.readBytes(rdata, width * 3);
                // shuffle rgb
                for (int i = 0; i < width; i++)
                {
                    mData[index + i] = ImagePixelUInt8x4(rdata[i * 3 + 2], rdata[i * 3 + 1], rdata[i * 3], 255);
                }

                index += width;
            }

            delete[] rdata;
        }
        else
        {
            LOG_DEBUG( LOG_IO, "%s: format not supported!", name);
            return false;
        }
    }

    return true;
}

template<> bool Image<IPF_uint8x4>::writeImage(const char *name)
{
    if (mData == 0)
    {
        return false;
    }

    System::File f;
    if (!f.open(name, System::File::FILE_WRITE))
    {
        LOG_DEBUG( LOG_IO, "error writing %s!\n", name);
        return false;
    }

    int scanSize = mWidth * 3;
    scanSize = (scanSize + 3) & ~0x3;
    int fileSize = scanSize * mHeight + sizeof(BITMAP_HEADER) + sizeof(BMPV5_INFO_HEADER);

    BITMAP_HEADER head;
    BMPV5_INFO_HEADER info;

    System::MemorySet(&head, 0, sizeof(BITMAP_HEADER));
    System::MemorySet(&info, 0, sizeof(BMPV5_INFO_HEADER));

    head.fileType = 0x4d42;
    head.fileSize = fileSize;
    head.bitmapPtr = sizeof(BITMAP_HEADER) + sizeof(BMPV5_INFO_HEADER);

    info.v3.bpp = 24;
    info.v3.width = mWidth;
    info.v3.height = mHeight;
    info.v3.pixelPerMeterY = info.v3.pixelPerMeterX = 0x0b12; // 72 DPI
    info.v3.headerSize = sizeof(BMPV5_INFO_HEADER);
    info.v3.bitplanesNum = 1;
    info.v3.size = scanSize * mHeight;
    info.colorSpace = 0x73524742; // sRGB color space

    f.writeBytes(&head, sizeof(BITMAP_HEADER));
    f.writeBytes(&info, sizeof(BMPV5_INFO_HEADER));

    uchar *rdata = new uchar[scanSize];
    System::MemorySet(rdata, 0, scanSize);

    for (int y = mHeight - 1; y >= 0; y--)
    {
        int index = y * mWidth;
        for (uint i = 0; i < mWidth; i++)
        {
            rdata[i * 3] = mData[index + i][0];
            rdata[i * 3 + 1] = mData[index + i][1];
            rdata[i * 3 + 2] = mData[index + i][2];
        }

        f.writeBytes(rdata, scanSize);
    }

    delete[] rdata;

    f.close();

    return true;
}

