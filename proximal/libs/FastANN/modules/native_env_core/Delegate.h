/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef DELEGATE_H_
#define DELEGATE_H_

namespace NVR {

// simple single argument delegate implementation
template<typename ReturnType, typename ArgType>
class Delegate
{
    typedef ReturnType (*StubPtrType)(void *instancePtr, ArgType arg);

public:
    Delegate()
            : mInstancePtr(0), mStubPtr(0)
    {
    }

    template<ReturnType (*TMethod)(ArgType)>
    static Delegate CreateFromFunction(void)
    {
        return Delegate(0, &FunctionStub<TMethod>);
    }

    template<class T, ReturnType (T::*TMethod)(ArgType)>
    static Delegate CreateFromMethod(T *instancePtr)
    {
        return Delegate(instancePtr, &MethodStub<T, TMethod>);
    }

    template<class T, ReturnType (T::*TMethod)(ArgType) const>
    static Delegate CreateFromConstMethod(T const *instancePtr)
    {
        return Delegate(const_cast<T*>(instancePtr), &ConstMethodStub<T, TMethod>);
    }

    ReturnType operator()(ArgType arg) const
    {
        return (*mStubPtr)(mInstancePtr, arg);
    }

    operator bool() const
    {
        return mStubPtr != 0;
    }

    bool operator!() const
    {
        return !(operator bool());
    }

private:
    Delegate(void *instancePtr, StubPtrType stubPtr)
            : mInstancePtr(instancePtr), mStubPtr(stubPtr)
    {
    }

    template<ReturnType (*TMethod)(ArgType)>
    static ReturnType FunctionStub(void *, ArgType arg)
    {
        return (TMethod)(arg);
    }

    template<class T, ReturnType (T::*TMethod)(ArgType)>
    static ReturnType MethodStub(void *instancePtr, ArgType arg)
    {
        T *p = static_cast<T*>(instancePtr);
        return (p->*TMethod)(arg);
    }

    template<class T, ReturnType (T::*TMethod)(ArgType) const>
    static ReturnType ConstMethodStub(void *instancePtr, ArgType arg)
    {
        T const *p = static_cast<T*>(instancePtr);
        return (p->*TMethod)(arg);
    }

    void *mInstancePtr;
    StubPtrType mStubPtr;
};

}

#endif /* DELEGATE_H_ */
