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
#ifdef ANDROID
#include <android/asset_manager.h>
#endif
#include "SystemCore.h"
#include "SystemFile.h"
#include "DeviceProxy.h"
#include "MemAlloc.h"

#define STD_FILE(x) (static_cast<FILE *>(x))
#define AAF_FILE(x) (static_cast<AAsset *>(x))

static System::BlockAllocator *GetFileCacheBlockAllocator(void)
{
    std::vector<managed_ptr<ManagedAbstractObject> > &components(System::Internal::DeviceProxy::GetComponents());

    System::BlockAllocator *instance =
            static_cast<System::BlockAllocator *>(components[System::Internal::DeviceProxy::ComponentFileCacheBlockAllocator]
                    .get());
    if (instance == 0)
    {
        instance = new System::BlockAllocator(FILE_CACHE_SIZE);
        components[System::Internal::DeviceProxy::ComponentFileCacheBlockAllocator] = instance;
    }

    return instance;
}

System::File::File(void)
        : mSource(0), mSize(0), mOffset(0), mCache(0), mCacheOffset(0), mCacheSize(0), mOpenMode(FILE_READ),
          mNativeAccessMode(0)
{
}

System::File::~File(void)
{
    close();
}

void System::File::reset(void)
{
    mSource = 0;
    mSize = 0;
    mOffset = 0;

    mCache = 0;
    mCacheOffset = 0;
    mCacheSize = 0;

    mOpenMode = FILE_READ;
    mNativeAccessMode = 0;
}

#ifdef ANDROID
static char const *SkipPrefix(char const *str, char const *prefix)
{
    int index = 0;
    while (str[index] == prefix[index] && str[index] != 0)
    {
        index++;
    }

    if (prefix[index] != 0)
    {
        return str;
    }

    return str + index;
}
#endif

/**
 * Checks whether a file exists in native file system.
 * @param name - file name
 * @return true if file exists, false otherwise
 */
bool System::File::Exists(const char *name)
{
    bool fileExists = false;

#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
    std::string fileName(Internal::DeviceProxy::GetInstance()->getExternalPath());
    fileName += name;
    FILE *stdFile = fopen(fileName.c_str(), "rb");
    if (stdFile)
    {
        fileExists = true;
        fclose(stdFile);
    }
#endif
#ifdef ANDROID
    else
    {
        // try to open as an asset in the .apk
        AAssetManager *am = Internal::DeviceProxy::GetInstance()->getNativeAssetManager();
        if (am != 0)
        {
            AAsset *apkFile = AAssetManager_open(am, SkipPrefix(name, "assets/"), AASSET_MODE_UNKNOWN);
            if (apkFile)
            {
                AAsset_close(apkFile);
                fileExists = true;
            }
        }
    }
#endif

    return fileExists;
}

/**
 * Opens a file from native file system.
 * @param name - file name
 * @param mode - file open mode: FILE_READ or FILE_WRITE
 * @return true if operation was success, false otherwise
 */
bool System::File::open(const char *name, FileOpenMode mode)
{
    if (mSource != 0)
    {
        // file already opened
        close();
    }

    uint nativeAccessMode = 0;
    void *nativeFile = 0;

    if (mode == FILE_WRITE)
    {
        // write
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
        std::string fileName(Internal::DeviceProxy::GetInstance()->getExternalPath());
        fileName += name;
        nativeFile = fopen(fileName.c_str(), "wb");
#endif
    }
    else
    {
#ifdef ANDROID
        // try to open as an asset in the .apk
        nativeAccessMode = 1;
        AAssetManager *am = Internal::DeviceProxy::GetInstance()->getNativeAssetManager();
        if (am != 0)
        {
            nativeFile = AAssetManager_open(am, SkipPrefix(name, "assets/"), AASSET_MODE_UNKNOWN);
        }

        if (!nativeFile)
        {
#endif
            // fread
            nativeAccessMode = 0;
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            std::string fileName(Internal::DeviceProxy::GetInstance()->getExternalPath());
            fileName += name;
            nativeFile = fopen(fileName.c_str(), "rb");

            if (!nativeFile)
            {
                nativeFile = fopen(name, "rb");
            }
#endif

#ifdef ANDROID
        }
#endif
    }

    if (!nativeFile)
    {
        return false;
    }

    mOpenMode = mode;
    mSource = nativeFile;
    mNativeAccessMode = nativeAccessMode;
    mCache = GetFileCacheBlockAllocator()->allocate();

    if (mode == FILE_WRITE)
    {
        mCacheOffset = 0;
        mCacheSize = 0;

        mOffset = 0;
        mSize = 0;
    }
    else
    {
        mCacheOffset = FILE_CACHE_SIZE;
        mCacheSize = FILE_CACHE_SIZE;

        mOffset = 0;
#ifdef ANDROID
        if (mNativeAccessMode == 0)
        {
#endif
#if defined(_WIN32)|| defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            fseek(STD_FILE(mSource), 0, SEEK_END);
            mSize = ftell(STD_FILE(mSource));
            fseek(STD_FILE(mSource), 0, SEEK_SET);
#endif
#ifdef ANDROID
        }
        else
        {
            mSize = AAsset_getLength(AAF_FILE(mSource));
        }
#endif
    }

    return true;
}

/**
 * Opens a file from memory location. This lets the programmer to load data with
 * single file loader directly from memory.
 * @param data - pointer to file data begining
 * @param size - data size in bytes
 * @return true if operation was success, false otherwise
 */
bool System::File::open(const uchar *data, uint size)
{
    mOpenMode = FILE_READ;
    mSource = const_cast<uchar *>(data);

    mCache = static_cast<uchar *>(mSource);
    mCacheOffset = 0;
    mCacheSize = size;

    mOffset = size;
    mSize = size;

    return true;
}

/**
 * Performs a cache flush, closes the file and frees all the resources associated with it.
 */
void System::File::close(void)
{
    if (mSource != mCache)
    {
        // physical file
        if (mOpenMode == FILE_WRITE)
        {
            if (mCacheOffset != 0)
            {
                // flush write cache
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
                fwrite(mCache, mCacheOffset, 1, STD_FILE(mSource));
#endif
            }
        }

#ifdef ANDROID
        if (mNativeAccessMode == 0)
        {
#endif
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            fclose(STD_FILE(mSource));
#endif
#ifdef ANDROID
        }
        else
        {
            AAsset_close(AAF_FILE(mSource));
        }
#endif

        // free cache block
        GetFileCacheBlockAllocator()->deallocate(mCache);

        // move to closed state
        reset();
    }
}

/**
 * Loads data from file system if local cache offset is set to a cache size value.
 * @return true(1) if cache update was a success, false(0) if EOF or other
 * error occured.
 */
bool System::File::cacheUpdate(void)
{
    if (mCacheOffset != mCacheSize)
    {
        return true;
    }

    if (mOffset == mSize)
    {
        return false;
    }

    int newSize = mSize - mOffset;
    if (newSize > FILE_CACHE_SIZE)
    {
        newSize = FILE_CACHE_SIZE;
    }

#ifdef ANDROID
    if (mNativeAccessMode == 0)
    {
#endif
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
        if (!fread(mCache, newSize, 1, STD_FILE(mSource)))
        {
            if (feof(STD_FILE(mSource)) == 0)
            {
                return false;
            }
        }
#endif
#ifdef ANDROID
    }
    else
    {
        int byteCount = newSize;
        uchar *destPtr = mCache;
        while (byteCount > 0)
        {
            int bytesRead = AAsset_read(AAF_FILE(mSource), destPtr, byteCount);
            if (bytesRead <= 0)
            {
                return false;
            }
            byteCount -= bytesRead;
            destPtr += bytesRead;
        }
    }
#endif

    mCacheOffset = 0;
    mCacheSize = newSize;
    mOffset += newSize;

    return true;
}

/**
 * Writes one byte to the file. Write is cached
 * @param value - byte value to be written
 */
void System::File::writeByte(int value)
{
    if (mOpenMode != FILE_WRITE)
    {
        return;
    }

    if (mCacheOffset == FILE_CACHE_SIZE)
    {
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
        fwrite(mCache, mCacheOffset, 1, STD_FILE(mSource));
#endif
        mOffset += FILE_CACHE_SIZE;
        mCacheOffset = 0;
    }

    mCache[mCacheOffset++] = value;
}

/**
 * Writes an array of bytes to the file. Write is cached
 * @param data - pointer to data
 * @param size - data size in bytes
 */
void System::File::writeBytes(const void *data, int size)
{
    if (mOpenMode != FILE_WRITE || size <= 0)
    {
        return;
    }

    while (mCacheOffset + size >= FILE_CACHE_SIZE)
    {
        uint count = FILE_CACHE_SIZE - mCacheOffset;
        MemoryCopy(&mCache[mCacheOffset], data, count);
        mCacheOffset += count;
        data = (char *)data + count;
        size -= count;

#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
        fwrite(mCache, mCacheOffset, 1, STD_FILE( mSource ));
#endif
        mCacheOffset = 0;
        mOffset += FILE_CACHE_SIZE;
    }

    MemoryCopy(&mCache[mCacheOffset], data, size);
    mCacheOffset += size;
}

/**
 * Reads a specified number of bytes from input file
 * @param data - destination array pointer
 * @param size - number of bytes to read
 * @return number of bytes read, 0 if read was unsuccessful
 */
uint System::File::readBytes(void *data, int size)
{
    if (mOpenMode != FILE_READ || size <= 0)
    {
        return 0;
    }

    // real file offset
    int offset = mOffset - mCacheSize + mCacheOffset;
    if (offset + size > mSize)
    {
        size = mSize - offset;
    }

    if (size == 0)
    {
        return 0;
    }

    uchar *dest = (uchar *)data;
    uint left = size;
    while (left > 0)
    {
        if (!cacheUpdate())
        {
            return 0;
        }

        uint count = mCacheSize - mCacheOffset;
        if (count > left)
        {
            count = left;
        }

        MemoryCopy(dest, &mCache[mCacheOffset], count);
        mCacheOffset += count;
        dest += count;
        left -= count;
    }

    return size;
}

/**
 * Reads a specified number of bytes from input file bypassing i/o cache. For file sources, after
 * call to this function cache will be invalid but can be updated with a call to System::File::CacheUpdate().
 * Remember that cache update or cached operations change the value of file pointer by multiplies of
 * FILE_CACHE_SIZE!
 * @param data - destination array pointer
 * @param size - number of bytes to read
 * @return number of bytes read, 0 if read was unsuccessful
 */
uint System::File::readBytesNoCache(void *data, int size)
{
    if (mOpenMode != FILE_READ || size <= 0)
    {
        return 0;
    }

    if (mSource == mCache)
    {
        // memory source
        if (mCacheOffset + size > mSize)
        {
            size = mSize - mCacheOffset;
        }

        if (size == 0)
        {
            return 0;
        }

        MemoryCopy(data, &mCache[mCacheOffset], size);
        mCacheOffset += size;
    }
    else
    {
        // file source

        // real file offset
        if (mOffset + size > mSize)
        {
            size = mSize - size;
        }

        if (size == 0)
        {
            return 0;
        }

#ifdef ANDROID
        if (mNativeAccessMode == 0)
        {
#endif
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            if (!fread(data, size, 1, STD_FILE(mSource)))
            {
                if (feof(STD_FILE(mSource)) == 0)
                {
                    return false;
                }
            }
#endif
#ifdef ANDROID
        }
        else
        {
            int byteCount = size;
            uchar *destPtr = static_cast<uchar *>(data);
            while (byteCount > 0)
            {
                int bytesRead = AAsset_read(AAF_FILE(mSource), destPtr, byteCount);
                if (bytesRead <= 0)
                {
                    return false;
                }
                byteCount -= bytesRead;
                destPtr += bytesRead;
            }
        }
#endif
        mOffset += size;
        mCacheOffset = mCacheSize;
    }

    return size;
}

/**
 * Reads specified number of bytes from input file. Read is aborted
 * and string terminated when new line character is read (0x0a) or maximum
 * string length is reached.
 * @param data - destination string pointer
 * @param maxLength - maximum string length before read termination
 * @return number of bytes read, 0 if read was unsuccessful
 */
uint System::File::readLine(void *data, int maxLength)
{
    if (maxLength <= 0)
    {
        return 0;
    }

    if (mOpenMode != FILE_READ)
    {
        return 0;
    }

    uchar *dest = (uchar *)data;
    int index = 0;
    maxLength--;
    while (index < maxLength)
    {
        if (!cacheUpdate())
        {
            break;
        }

        dest[index++] = mCache[mCacheOffset++];

        if (dest[index - 1] == '\n')
        {
            break;
        }
    }

    if (index == 0)
    {
        return 0;
    }

    dest[index++] = 0;
    return index;
}

/**
 * Changes the position of file marker.
 * @param offset - file offset position
 * @param mode - offset interpretation mode (SEEK_FROM_START, SEEK_FROM_CURRENT, SEEK_FROM_END)
 * @return true if seek was a success, false otherwise
 */
bool System::File::seek(int offset, FileSeekMode mode)
{
    int noffset;

    if (mOpenMode != FILE_READ || mSource == 0)
    {
        return false;
    }

    if (mode == SEEK_FROM_CURRENT)
    {
        noffset = mOffset - mCacheSize + mCacheOffset + offset;
    }
    else if (mode == SEEK_FROM_END)
    {
        noffset = mSize - offset;
    }
    else
    {
        noffset = offset;
    }

    if (noffset < 0 || noffset > mSize)
    {
        return false;
    }

    if (noffset > mOffset || noffset < mOffset - mCacheSize)
    {
        // out of scope, seek
        mCacheOffset = mCacheSize;
        mOffset = noffset;

#ifdef ANDROID
        if (mNativeAccessMode == 0)
        {
#endif
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            fseek(STD_FILE(mSource), noffset, SEEK_SET);
#endif
#ifdef ANDROID
        }
        else
        {
            AAsset_seek(AAF_FILE(mSource), noffset, SEEK_SET);
        }
#endif
    }
    else
    {
        // still in cache
        mCacheOffset = noffset - mOffset + mCacheSize;
    }

    return true;
}

/**
 * Changes the position of file marker while bypassing the i/o cache. For file sources, after
 * call to this function cache will be invalid but can be updated with a call to cacheUpdate()
 * Remember that cache update or cached operations change the value of file pointer by multiplies of
 * FILE_CACHE_SIZE!
 * @param offset - file offset position
 * @param mode - offset interpretation mode (SEEK_FROM_START, SEEK_FROM_CURRENT, SEEK_FROM_END)
 * @return true if seek was a success, false otherwise
 */
bool System::File::seekNoCache(int offset, FileSeekMode mode)
{
    int noffset;

    if (mOpenMode != FILE_READ || mSource == 0)
    {
        return false;
    }

    if (mode == SEEK_FROM_CURRENT)
    {
        if (mSource == mCache)
        {
            // memory source
            noffset = mCacheOffset + offset;
        }
        else
        {
            // file source
            noffset = mOffset + offset;
        }
    }
    else if (mode == SEEK_FROM_END)
    {
        noffset = mSize - offset;
    }
    else
    {
        noffset = offset;
    }

    if (noffset < 0 || noffset > mSize)
    {
        return false;
    }

    if (mSource == mCache)
    {
        // memory source, cache IS the file
        mCacheOffset = noffset;
    }
    else
    {
        // file source, seek
        mCacheOffset = mCacheSize;
        mOffset = noffset;
#ifdef ANDROID
        if (mNativeAccessMode == 0)
        {
#endif
#if defined(_WIN32) || defined(LINUX) || defined(ANDROID) || (defined(__APPLE__) && defined(__MACH__))
            fseek(STD_FILE(mSource), noffset, SEEK_SET);
#endif
#ifdef ANDROID
        }
        else
        {
            AAsset_seek(AAF_FILE(mSource), noffset, SEEK_SET);
        }
#endif
    }

    return true;
}

/**
 * Gets the file marker position (from file beginning)
 * @return file marker position from file begin
 */
uint System::File::tell(void)
{
    if (mOpenMode != FILE_READ)
    {
        return mOffset + mCacheOffset;
    }
    else
    {
        return mOffset - mCacheSize + mCacheOffset;
    }
}

/**
 * Gets the file marker position (from file beginning)
 * bypassing i/o cache. When using only cached reads/writes the true
 * file marker will show values which are multiplies of FILE_CACHE_SIZE
 * @return file marker position from file begin
 */
uint System::File::tellNoCache(void)
{
    if (mSource == mCache)
    {
        // memory source, return offset in cache
        return mCacheOffset;
    }
    return mOffset;
}

