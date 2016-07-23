/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef MANAGEDPTR_H_
#define MANAGEDPTR_H_

/**
 * Base class for all objects managed with reference counting
 */
template<typename T> class ManagedObject
{
public:
    ManagedObject(void)
            : mRefCount(0)
    {
    }

    friend void managed_ptr_acquire(T *ptr)
    {
        ptr->mRefCount++;
    }

    friend void managed_ptr_release(T *ptr)
    {
        ptr->mRefCount--;
        if (ptr->mRefCount <= 0)
        {
            delete ptr;
        }
    }

private:
    int mRefCount;
};

class ManagedAbstractObject
{
public:
    ManagedAbstractObject(void)
            : mRefCount(0)
    {
    }

    virtual ~ManagedAbstractObject(void) = 0;

    friend void managed_ptr_acquire(ManagedAbstractObject *ptr)
    {
        ptr->mRefCount++;
    }

    friend void managed_ptr_release(ManagedAbstractObject *ptr)
    {
        ptr->mRefCount--;
        if (ptr->mRefCount <= 0)
        {
            delete ptr;
        }
    }

private:
    int mRefCount;
};

/**
 * Managed pointer template
 */

template<typename T> class managed_ptr
{
    template<typename U> friend class managed_ptr;
public:
    managed_ptr(void)
            : mPtr(0)
    {
    }

    managed_ptr(T *p, bool acquire = true)
            : mPtr(p)
    {
        if (mPtr != 0 && acquire)
        {
            managed_ptr_acquire(mPtr);
        }
    }

    template<typename U> managed_ptr(managed_ptr<U> const &rhs)
            : mPtr(static_cast<T *>(rhs.mPtr))
    {
        if (mPtr != 0)
        {
            managed_ptr_acquire(mPtr);
        }
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    template<typename U> managed_ptr( managed_ptr<U> &&rhs )
    : mPtr( static_cast<T *>( rhs.mPtr ) )
    {
        rhs.mPtr = 0;
    }
#endif

    managed_ptr(managed_ptr const &rhs)
            : mPtr(rhs.mPtr)
    {
        if (mPtr != 0)
        {
            managed_ptr_acquire(mPtr);
        }
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    managed_ptr( managed_ptr &&rhs )
    : mPtr( rhs.mPtr )
    {
        rhs.mPtr = 0;
    }
#endif

    ~managed_ptr(void)
    {
        if (mPtr != 0)
        {
            managed_ptr_release(mPtr);
            mPtr = 0;
        }
    }

    T *get(void) const
    {
        return mPtr;
    }

    void reset(void)
    {
        if (mPtr != 0)
        {
            managed_ptr_release(mPtr);
            mPtr = 0;
        }
    }

    void swap(managed_ptr &rhs)
    {
        T *tmp = mPtr;
        mPtr = rhs.mPtr;
        rhs.mPtr = tmp;
    }

    managed_ptr &operator=(managed_ptr const &rhs)
    {
        if (rhs.mPtr != 0)
        {
            managed_ptr_acquire(rhs.mPtr);
        }

        if (mPtr != 0)
        {
            managed_ptr_release(mPtr);
        }

        mPtr = rhs.mPtr;

        return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    managed_ptr &operator=( managed_ptr &&rhs )
    {
        T *tmp = mPtr;
        mPtr = rhs.mPtr;
        rhs.mPtr = tmp;

        return *this;
    }
#endif

    T &operator*(void) const
    {
        return *mPtr;
    }

    T *operator->(void) const
    {
        return mPtr;
    }

    operator bool(void)
    {
        return mPtr == 0;
    }

private:

    T *mPtr;
};

template<typename T, typename U> inline bool operator==(managed_ptr<T> const &a, managed_ptr<U> const &b)
{
    return a.get() == b.get();
}

template<typename T, typename U> inline bool operator!=(managed_ptr<T> const &a, managed_ptr<U> const &b)
{
    return a.get() != b.get();
}

template<typename T, typename U> inline bool operator==(managed_ptr<T> const &a, U *b)
{
    return a.get() == b;
}

template<typename T, typename U> inline bool operator!=(managed_ptr<T> const &a, U *b)
{
    return a.get() != b;
}

template<typename T, typename U> inline bool operator==(T *a, managed_ptr<U> const &b)
{
    return a == b.get();
}

template<typename T, typename U> inline bool operator!=(T *a, managed_ptr<U> const &b)
{
    return a != b.get();
}

template<typename T> inline bool operator<(managed_ptr<T> const &a, managed_ptr<T> const &b)
{
    return a.get() < b.get();
}

#endif /* MANAGEDPTR_H_ */
