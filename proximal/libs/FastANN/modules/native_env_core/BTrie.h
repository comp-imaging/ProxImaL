/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _BTRIE_H
#define _BTRIE_H

#include "Base.h"

namespace NVR
{

template<typename T> class BTrieNode
{
public:
    typedef uint KeyType;
    typedef uint KeyTypeRef;

    enum
    {
        KeyBucketCount = sizeof(KeyType) * 8
    };

    BTrieNode(void)
            : mKey(0)
    {
    }

    BTrieNode(KeyType const &key)
            : mKey(key)
    {
    }

    ~BTrieNode(void)
    {
    }

    FORCE_INLINE KeyTypeRef key(void)
    {
        return mKey;
    }

    FORCE_INLINE KeyTypeRef const key(void) const
    {
        return mKey;
    }

    FORCE_INLINE T* &child(int num)
    {
        return mChild[num];
    }

    FORCE_INLINE T* const &child(int num) const
    {
        return mChild[num];
    }

    FORCE_INLINE static int GetKeyBucketIndex(KeyType const &key)
    {
        return key == 0 ? (KeyBucketCount - 1) : __builtin_clz(key);
    }

protected:
    KeyType mKey;
    T *mChild[2];
};

template<typename T> class BTrieNodeSimple
{
public:
    typedef uint KeyType;
    typedef uint KeyTypeRef;

    enum
    {
        KeyBucketCount = 1
    };

    BTrieNodeSimple(void)
            : mKey(0)
    {
    }

    BTrieNodeSimple(KeyType const &key)
            : mKey(key)
    {
    }

    ~BTrieNodeSimple(void)
    {
    }

    FORCE_INLINE KeyTypeRef key(void)
    {
        return mKey;
    }

    FORCE_INLINE KeyTypeRef const key(void) const
    {
        return mKey;
    }

    FORCE_INLINE T* &child(int num)
    {
        return mChild[num];
    }

    FORCE_INLINE T* const &child(int num) const
    {
        return mChild[num];
    }

    FORCE_INLINE static int GetKeyBucketIndex(KeyTypeRef key)
    {
        (void)key;
        return 0;
    }

protected:
    KeyType mKey;
    T *mChild[2];
};

template<typename T> class BTrie
{
    typedef typename T::KeyType KeyType;
    typedef typename T::KeyTypeRef KeyTypeRef;
public:
    BTrie(void)
    {
        for (int i = 0; i < T::KeyBucketCount; i++)
        {
            mBuckets[i] = 0;
        }
    }

    T *insert(T *node)
    {
        KeyTypeRef nodeKey(node->key());
        int bindex = T::GetKeyBucketIndex(nodeKey);
        KeyType mask = static_cast<KeyType>(1) << ((T::KeyBucketCount - 1) - bindex);
        T **parentNode = &mBuckets[bindex];
        T *currentNode = *parentNode;

        while (currentNode != 0)
        {
            if (currentNode->key() == nodeKey)
            {
                return currentNode;
            }
            mask >>= 1;
            int cnum = (nodeKey & mask) != 0;
            parentNode = &currentNode->child(cnum);
            currentNode = *parentNode;
        }

        *parentNode = node;
        node->child(0) = 0;
        node->child(1) = 0;
        return node;
    }

    T *remove(KeyTypeRef key)
    {
        int bindex = T::GetKeyBucketIndex(key);
        KeyType mask = static_cast<KeyType>(1) << ((T::KeyBucketCount - 1) - bindex);
        T **parentNode = &mBuckets[bindex];
        T *currentNode = *parentNode;

        while (currentNode != 0)
        {
            if (currentNode->key() == key)
            {
                break;
            }
            mask >>= 1;
            int cnum = (key & mask) != 0;
            parentNode = &currentNode->child(cnum);
            currentNode = *parentNode;
        }

        if (currentNode == 0)
        {
            return 0;
        }

        // replace removed node with any leaf node
        T **pgnode = parentNode;
        T *gnode = currentNode;
        for (;;)
        {
            // TODO: investigate how child removal priority (left or right) affects trie quality
            if (gnode->child(0) != 0)
            {
                pgnode = &gnode->child(0);
                gnode = *pgnode;
            }
            else if (gnode->child(1) != 0)
            {
                pgnode = &gnode->child(1);
                gnode = *pgnode;
            }
            else
            {
                break;
            }
        }

        *pgnode = 0;
        if (gnode != currentNode)
        {
            *parentNode = gnode;
            gnode->child(0) = currentNode->child(0);
            gnode->child(1) = currentNode->child(1);
        }

        return currentNode;
    }

    T *get(KeyTypeRef key)
    {
        int bindex = T::KeyBucketIndex(key);
        KeyType mask = static_cast<KeyType>(1) << ((T::KeyBucketCount - 1) - bindex);
        T *currentNode = mBuckets[bindex];

        while (currentNode != 0)
        {
            if (currentNode->key() == key)
            {
                return currentNode;
            }
            mask >>= 1;
            int cnum = (key & mask) != 0;
            currentNode = currentNode->child(cnum);
        }

        return 0;
    }

    T *getMinimum(void)
    {
        T *currentNode;
        for (int i = T::KeyBucketCount - 1; i >= 0; i--)
        {
            currentNode = mBuckets[i];
            if (currentNode != 0)
            {
                break;
            }
        }

        if (currentNode == 0)
        {
            return 0;
        }

        KeyTypeRef candidateKey(currentNode->key());
        T *candidateNode = currentNode;
        currentNode = currentNode->child(currentNode->child(0) == 0);

        while (currentNode != 0)
        {
            KeyTypeRef currentKey(currentNode->key());
            if (currentKey < candidateKey)
            {
                candidateKey = currentKey;
                candidateNode = currentNode;
            }

            currentNode = currentNode->child(currentNode->child(0) == 0);
        }

        return candidateNode;
    }

    T *getMaximum(void)
    {
        T *currentNode;
        for (int i = 0; i < T::KeyBucketCount; i++)
        {
            currentNode = mBuckets[i];
            if (currentNode != 0)
            {
                break;
            }
        }

        if (currentNode == 0)
        {
            return 0;
        }

        KeyTypeRef candidateKey(currentNode->key());
        T *candidateNode = currentNode;
        currentNode = currentNode->child(currentNode->child(1) == 0);

        while (currentNode != 0)
        {
            KeyTypeRef currentKey(currentNode->key());
            if (currentKey > candidateKey)
            {
                candidateKey = currentKey;
                candidateNode = currentNode;
            }

            currentNode = currentNode->child(currentNode->child(1) == 0);
        }

        return candidateNode;
    }

private:
    T *mBuckets[T::KeyBucketCount];
};

}

#endif

