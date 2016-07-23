/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifdef ANDROID
#include <android/log.h>
#define ANDROID_MODULE_NAME "nv_rcf"
#endif
#include <stdio.h>
#ifdef LINUX
#include <wchar.h>
#endif
#include <string.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <stdarg.h>
#include "SystemCore.h"

// debug log level mask
static uint g_DebugLogLevelMask = LOG_ALL;

// ==========================================================================
// MEMORY
// ==========================================================================

void *System::Internal::MemoryAlloc(size_t size, size_t alignSize)
{
    void *ptr = malloc(size + alignSize + sizeof(void *));
    if (ptr == 0)
    {
        return 0;
    }

    void **aptr = (void **)(((size_t)ptr + sizeof(void *) + (alignSize - 1)) & ~(alignSize - 1));
    aptr[-1] = ptr;
    return aptr;
}

void System::MemoryFree(void *ptr)
{
    if (ptr != 0)
    {
        free(((void **)ptr)[-1]);
    }
}

/**
 * Sets the memory block to specific value
 * @param dest - destination block pointer
 * @param value - byte value to be set
 * @param size - block size in bytes
 */
void System::MemorySet(void *dest, int value, int size)
{
    memset(dest, value, size);
}

/**
 * Copies the memory block. When blocks overlap behaviour is undefined.
 * @param dest - destination block pointer
 * @param dest - source block pointer
 * @param size - block size in bytes
 */
void System::MemoryCopy(void *dest, const void *src, int size)
{
    memcpy(dest, src, size);
}

/**
 * Copies the memory block. Block overlapping is resolved properly.
 * @param dest - destination block pointer
 * @param dest - source block pointer
 * @param size - block size in bytes
 */
void System::MemoryMove(void *dest, const void *src, int size)
{
    memmove(dest, src, size);
}

// simple SDBM hash function
#define HASH_BYTE(hash,c) ((c)+((hash)<<6)+((hash)<<16)-(hash))

/**
 * Calculated 32-bit hash value from memory data
 * @param data - pointer to memory block
 * @param length - block size in bytes
 */
uint System::MemoryHash(const void *data, int length)
{
    const uchar *str = (const uchar *)data;
    uint hash = 0;

    while (length != 0)
    {
        int c = *str++;
        hash = HASH_BYTE(hash, c);
        length--;
    }

    return hash;
}

// ==========================================================================
// STRING
// ==========================================================================

/**
 * Gets the 32-bit string hash value.
 * @param data - pointer to null terminated string
 * @return string hash value
 */
uint System::StringHash(const char *data)
{
    if (!data)
    {
        return 0;
    }

    const uchar *str = (const uchar *)data;
    uint hash = 0;
    int c;

    while ((c = *str) != 0)
    {
        hash = HASH_BYTE(hash, c);
        str++;
    }

    return hash;
}

/**
 * Gets the 32-bit string hash value.
 * @param data - pointer to null terminated string
 * @param length - maximum chars to hash
 * @return string hash value
 */
uint System::StringHash(const char *data, int length)
{
    if (!data)
    {
        return 0;
    }

    const uchar *str = (const uchar *)data;
    uint hash = 0;
    int index = 0;
    int c;

    while ((c = str[index]) != 0 && index < length)
    {
        hash = HASH_BYTE(hash, c);
        index++;
    }

    return hash;
}

/**
 * Copies null terminated string to other memory location.
 * @param dest - destination location pointer
 * @param src - source string
 */
void System::StringCopy(char *dest, const char *src)
{
    if (!dest || !src)
    {
        return;
    }

    do
    {
        *dest++ = *src;
    }while (*src++ != 0);
}

/**
 * Appends a string to the end of other string. It's up to user to provide enough
 * place in dest for the new incoming string data.
 * @param dest - destination string location pointer
 * @param src - source string
 */
void System::StringAppend(char *dest, const char *src)
{
    if (!dest || !src)
    {
        return;
    }

    while (*dest != 0)
    {
        dest++;
    }

    do
    {
        *dest++ = *src;
    }while (*src++ != 0);
}

/**
 * Finds specific character within the input string and returns the pointer to it. If
 * character is not found null is returned.
 * @param src - source string
 * @param ch - searched character
 * @return pointer to the character or null if character was not found
 */
const char *System::StringFindChar(const char *src, char ch)
{
    while (*src != 0 && *src != ch)
    {
        src++;
    }

    if (*src == ch)
    {
        return (char *)src;
    }

    return 0;
}

/**
 * Finds last instance of a specific character within the input string and returns the pointer to it. If
 * character is not found null is returned.
 * @param str - source string
 * @param ch - searched character
 * @return pointer to the character or null if character was not found
 */
const char *System::StringFindCharLast(const char *str, char ch)
{
    int index = 0, lindex = -1;

    while (str[index] != 0)
    {
        if (str[index] == ch)
        {
            lindex = index;
        }
        index++;
    }

    if (lindex < 0)
    {
        return 0;
    }

    return &str[lindex];
}

/**
 * Compares two string
 * @param str1 - first source string
 * @param str2 - second source string
 * @return true if two strings are equal, false otherwise
 */
bool System::StringCompare(const char *str1, const char *str2)
{
    if (!str1 || !str2)
    {
        return false;
    }

    while (*str1 != 0 && *str1 == *str2)
    {
        str1++;
        str2++;
    }

    if (*str1 == *str2)
    {
        return true;
    }

    return false;
}

/**
 * Compares two string at certain distance from begining.
 * @param str1 - first source string
 * @param str2 - second source string
 * @param length - maximum number of characters to compare
 * @return true if two strings are equal, false otherwise
 */
bool System::StringCompare(const char *str1, const char *str2, int length)
{
    if (!str1 || !str2)
    {
        return false;
    }

    while (length > 0 && *str1 != 0 && *str1 == *str2)
    {
        str1++;
        str2++;
        length--;
    }

    if (length == 0)
    {
        return true;
    }

    return false;
}

/**
 * Converts string into an integer value
 * @param value - pointer to variable where integer number will be written to
 * @param str - string pointer holding integer number representation (base 10 number)
 * @return if successful an index to whitespace character following the parsed number, 0 otherwise
 */

int System::StringToInt(int *value, const char *str)
{
    if (str == 0)
    {
        return 0;
    }

    // skip white space
    int index = 0;
    while (str[index] == 0x20 || str[index] == 0x09 || str[index] == 0x0d || str[index] == 0x0a)
    {
        index++;
    }

    bool negative = false;

    if (str[index] == '-')
    {
        // negative sign found
        negative = true;
        index++;
    }

    int num = 0;
    while (str[index] >= '0' && str[index] <= '9')
    {
        num = (num << 3) + (num << 1) + (str[index] - '0');
        index++;
    }

    if (str[index] != 0 && str[index] != 0x20 && str[index] != 0x09 && str[index] != 0x0d && str[index] != 0x0a)
    {
        return 0;
    }

    // set value
    *value = negative ? -num : num;
    return index;
}

/**
 * Gets the string length
 * @param str - pointer to the string
 * @return string length
 */
int System::StringLength(const char *str)
{
    if (!str)
    {
        return 0;
    }

    int length = 0;

    while (*str++ != 0)
    {
        length++;
    }

    return length;
}

// ==========================================================================
// I/O
// ==========================================================================

/**
 * Enables/disables output of selected log streams
 * @param level - bit mask containing enabled active streams, LOG_*
 */
void System::LogMaskSet(uint level)
{
    g_DebugLogLevelMask = level;
}

/**
 * Prints debug info to specific output streams with accompanying source file information
 * @param level - log level bit mask, can create one from LOG_* constants.
 * @param file - source file name
 * @param line - source line number at which the function was invoked
 * @param func - function name
 * @param str - pointer to a string
 */
void System::DebugLog(uint level, const char *file, int line, const char *func, const char *str, ...)
{
    va_list va;
    va_start( va, str);

    int clevel = level & g_DebugLogLevelMask;
    if (clevel != 0)
    {
#if defined(_WIN32) || defined(LINUX)
        printf("[%s:%i] %s(): ", file, line, func);
        vprintf(str, va);
        printf("\n");
        fflush(stdout);
#endif
#ifdef ANDROID
        int const maxBufferLength = 1024;
        int androidLevel;
        char buffer[maxBufferLength];

        switch (clevel)
        {
            case LOG_AUX0:
            case LOG_AUX1:
            case LOG_AUX2:
            case LOG_AUX3:
            androidLevel = ANDROID_LOG_VERBOSE;
            break;
            default:
            androidLevel = ANDROID_LOG_INFO;
        }

        if (vsnprintf(buffer, maxBufferLength, str, va) >= 0)
        {
            __android_log_print(androidLevel, ANDROID_MODULE_NAME, "[%s:%i] %s(): %s", file, line, func, buffer);
        }
#endif

    }

    if (BIT_ISSET( level, 1 ))
    {
        DEBUG_BREAK();
    }

    va_end( va);
}

/**
 * Prints debug info on the console
 * @param str - pointer to a string
 */
void System::NativeLog(const char *str)
{
#if defined(_WIN32) || defined(LINUX)
    puts(str);
    puts("\n");
    fflush(stdout);
#endif

#ifdef ANDROID
    __android_log_write(ANDROID_LOG_INFO, ANDROID_MODULE_NAME, str);
#endif
}

/**
 * Prints debug info on the console
 * @param str - pointer to a string
 */
void System::NativeLog(const ushort *str)
{
#if defined(_WIN32) || defined(LINUX)
    wprintf(L"%s\n", str);
    fflush(stdout);
#else
    (void)str;
#endif
}

