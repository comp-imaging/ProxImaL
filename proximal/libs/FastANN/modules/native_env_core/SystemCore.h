/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef SYSTEMCORE_H_
#define SYSTEMCORE_H_

#include <stddef.h>
#include "Base.h"

// logging and debug
#define LOG_SYS  0x002
#define LOG_MEM  0x004
#define LOG_IO   0x008
#define LOG_LIB  0x010
#define LOG_INFO 0x020
#define LOG_AUX0 0x040
#define LOG_AUX1 0x080
#define LOG_AUX2 0x100
#define LOG_AUX3 0x200
#define LOG_ALL  0x3fe

#ifdef DEBUG_MODE
#if defined(_MSC_VER)
#define DEBUG_BREAK() (__debugbreak())
#else
#define DEBUG_BREAK() (__builtin_trap())
#endif
#define LOG(msg,...) (System::DebugLog(LOG_INFO,__FILE__,__LINE__,__FUNCTION__,(msg),##__VA_ARGS__))
#define LOG_DEBUG(level,msg,...) (System::DebugLog((level),__FILE__,__LINE__,__FUNCTION__,(msg),##__VA_ARGS__))
#define LOG_DEBUG_TRAP(level,msg,...) (System::DebugLog((level)|1,__FILE__,__LINE__,__FUNCTION__,(msg),##__VA_ARGS__))
#else
#define DEBUG_BREAK()
#define LOG(msg,...)
#define LOG_DEBUG(level,msg,...)
#define LOG_DEBUG_TRAP(level,msg,...)
#endif

namespace System
{

// MEMORY
namespace Internal
{
void *MemoryAlloc(size_t size, size_t alignSize = 1);
}
void MemoryFree(void *ptr);

void MemorySet(void *dest, int value, int size);
void MemoryCopy(void *dest, const void *src, int size);
void MemoryMove(void *dest, const void *src, int size);
uint MemoryHash(const void *data, int length);

template<typename T> FORCE_INLINE T *MemoryAlloc(size_t size, size_t alignSize = 1)
{
    return static_cast<T *>(Internal::MemoryAlloc(sizeof(T) * size, alignSize));
}

// STRING

// 8-bit characters
uint StringHash(const char *str);
uint StringHash(const char *str, int length);
void StringCopy(char *dest, const char *src);
void StringAppend(char *dest, const char *src);
const char *StringFindChar(const char *src, char ch);
const char *StringFindCharLast(const char *src, char ch);
bool StringCompare(const char *str1, const char *str2);
bool StringCompare(const char *str1, const char *str2, int length);
int StringLength(const char *str);
int StringToInt(int *value, const char *str);

// I/O
void DebugLog(uint level, const char *file, int line, const char *func, const char *str, ...);
void LogMaskSet(uint level);
void NativeLog(const char *str);
void NativeLog(const ushort *str);

}

#endif /* ABSTRACTOS_H_ */
