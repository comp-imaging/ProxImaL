/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef SYSTEMFILE_H_
#define SYSTEMFILE_H_

#include "Base.h"
#include "ManagedPtr.h"

namespace System
{

// file cache size per file
#define FILE_CACHE_SIZE  4096

// system file structure
class File: public ManagedObject<File>
{
public:
    // open modes
    enum FileOpenMode
    {
        FILE_READ, FILE_WRITE
    };

    enum FileSeekMode
    {
        // seek from file beginning
        SEEK_FROM_START,
        // seek from file current position
        SEEK_FROM_CURRENT,
        // seek from file end
        SEEK_FROM_END
    };

    File(void);
    ~File(void);

    bool open(const char *name, FileOpenMode mode);
    bool open(const uchar *data, uint size);

    void close(void);

    void writeByte(int value);

    void writeBytes(const void *data, int size);

    bool cacheUpdate(void);

    int readByte(void)
    {
        if (mOpenMode == FILE_WRITE)
        {
            return -1;
        }

        if (!cacheUpdate())
        {
            return -1;
        }

        return mCache[mCacheOffset++];
    }

    uint readBytes(void *data, int size);
    uint readBytesNoCache(void *data, int size);

    uint readLine(void *data, int maxLength);

    /**
     * Gets the file cache data pointer. Value might be different on each call!
     * @return pointer to file cache
     */
    uchar *getCache(void)
    {
        return mCache + mCacheOffset;
    }

    /**
     * Gets the file cache data size. Value might be different on each call!
     * @return file cache size in bytes
     */
    uint getCacheSize(void)
    {
        return mCacheSize - mCacheOffset;
    }

    /**
     * Gets the file data size.
     * @return file size in bytes
     */
    uint getSize(void)
    {
        return mSize;
    }

    bool seek(int offset, FileSeekMode mode);
    bool seekNoCache(int offset, FileSeekMode mode);
    uint tell(void);
    uint tellNoCache(void);

    static bool Exists(const char *name);

private:
    // prevent copy construction and assignment
    File(File const &instance);
    File &operator=(File const &instance);

    void reset(void);

    // source
    void *mSource;
    int mSize, mOffset;

    // internal cache
    uchar *mCache;
    int mCacheOffset, mCacheSize;
    FileOpenMode mOpenMode;
    uint mNativeAccessMode;
};

}

#endif /* SYSTEMFILE_H_ */
